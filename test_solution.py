import tempfile
from pathlib import Path

import database

database.DB_PATH = Path(tempfile.mkdtemp()) / "test_shop.db"
database.init_db()

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import solution
from critic import score_reply, PASS_SCORE

GOOD_REPLY = (
    "مشتری گرامی، از اینکه این مشکل برای شما پیش آمده صمیمانه عذرخواهی می‌کنیم و ناراحتی شما را "
    "کاملاً درک می‌کنیم. درخواست شما با کد پیگیری ۴۵۸۲ ثبت شد و برای بررسی تراکنش به واحد مالی "
    "ارجاع داده شد. در صورتی که سفارش ثبت نشده باشد، مبلغ طبق روال بانکی به حساب شما بازگردانده "
    "می‌شود. نتیجهٔ بررسی حداکثر تا ۴۸ ساعت آینده از طریق پیامک به شما اطلاع داده خواهد شد. "
    "از صبر و همراهی شما سپاسگزاریم."
)

NO_ACTION_REPLY = (
    "مشتری گرامی، از اینکه این مشکل برای شما پیش آمده صمیمانه عذرخواهی می‌کنیم و ناراحتی شما را "
    "کاملاً درک می‌کنیم. می‌دانیم که کم شدن مبلغ از حساب بدون ثبت سفارش چقدر می‌تواند نگران‌کننده "
    "باشد و حق دارید که از این وضعیت ناراضی باشید. از صبر و همراهی شما در این مدت سپاسگزاریم و "
    "امیدواریم بتوانیم اعتماد شما را دوباره به دست بیاوریم."
)

QUESTION_REPLY = (
    "مشتری گرامی، تا ۷ روز پس از تحویل، در صورت سالم بودن کالا و بسته‌بندی، امکان مرجوعی وجود دارد. "
    "برای ثبت مرجوعی کافی است شماره سفارش خود را برای پشتیبانی ارسال کنید."
)

BAD_REPLY = "مشکل شما قطعاً همین الان حل می‌شود."


def answer(text):
    return lambda *args: {"found": True, "answer": text}


NOT_FOUND = lambda *args: {"found": False, "answer": ""}


def classifier(intent, order_id=None, phone=None):
    return lambda *args: {"intent": intent, "order_id": order_id, "phone": phone}


ALI = database.get_user("ali")
MARYAM = database.get_user("maryam")
MOHAMMAD = database.get_user("mohammad")


def cart(*items):
    return {"items": [{"product_id": pid, "quantity": qty} for pid, qty in items]}


def sequence(*results):
    results = list(results)
    return lambda *args: results.pop(0)


def build_graph(draft_fn=answer(GOOD_REPLY), revise_fn=lambda *a: GOOD_REPLY, classify_fn=classifier("complaint"),
                extract_fn=lambda *a: cart(), confirm_fn=lambda *a: "confirm"):
    solution.generate_draft = draft_fn
    solution.revise_draft = revise_fn
    solution.classify_message = classify_fn
    solution.extract_order = extract_fn
    solution.interpret_confirmation = confirm_fn
    solution.retrieve = lambda query: ["سؤال متداول: نمونه"]
    return solution.create_complaint_graph(InMemorySaver())


def run(graph, thread_id, message="سه روز است مبلغ از حسابم کم شده اما سفارش ثبت نشده", user=None):
    config = solution.make_config(thread_id)
    graph.invoke(solution.initial_state(message, "test", user), config)
    return config, graph.get_state(config)


def test_critic_is_deterministic_and_rewards_good_reply():
    assert score_reply(GOOD_REPLY) == score_reply(GOOD_REPLY)
    score, feedback = score_reply(GOOD_REPLY)
    assert score == 100, feedback


def test_reply_without_action_never_gets_full_score():
    score, feedback = score_reply(NO_ACTION_REPLY)
    assert score < 100
    assert score < PASS_SCORE
    assert any("action" in item for item in feedback)


def test_forbidden_promise_is_penalized():
    score, feedback = score_reply(GOOD_REPLY + " بازگشت وجه را تضمین می‌کنیم.")
    assert score == 90
    assert any("promises" in item for item in feedback)


def test_question_answer_does_not_need_apology():
    score, feedback = score_reply(QUESTION_REPLY, "question")
    assert score == 100, feedback
    assert score_reply(QUESTION_REPLY, "complaint")[0] < PASS_SCORE


def test_retrieve_always_includes_full_catalog():
    import rag
    rag.embed_texts = lambda texts: [[1.0, float(i)] for i, _ in enumerate(texts)]
    rag.reset_index()

    context = rag.retrieve("گوشی آیفون دارید؟")
    assert context[0].startswith("فهرست کامل دسته‌بندی‌های فروشگاه")
    assert all(p["name"] in context[0] for p in database.get_products())
    assert len(context) == rag.TOP_K + 1


def test_good_revision_goes_to_finalize_and_is_approved():
    graph = build_graph(answer(BAD_REPLY), lambda *a: GOOD_REPLY)
    _, snapshot = run(graph, "t-approved")
    state = snapshot.values
    assert state["revision_count"] == 1
    assert state["score"] >= PASS_SCORE
    assert state["status"] == "approved"
    assert state["history"][-1] == "finalize: approved"
    assert not snapshot.interrupts


def test_revisions_never_exceed_two_and_human_review_is_requested():
    graph = build_graph(answer(BAD_REPLY), lambda *a: BAD_REPLY)
    config, snapshot = run(graph, "t-human")
    state = snapshot.values
    assert state["revision_count"] == solution.MAX_REVISIONS
    assert state["status"] == "needs_human_review"
    assert snapshot.interrupts[0].value["kind"] == "reply_review"

    graph.invoke(Command(resume={"action": "approve"}), config)
    state = graph.get_state(config).values
    assert state["status"] == "approved"
    assert state["operator_decision"] == "approved"


def test_operator_reject_keeps_needs_human_review():
    graph = build_graph(answer(BAD_REPLY), lambda *a: BAD_REPLY)
    config, _ = run(graph, "t-reject")
    graph.invoke(Command(resume={"action": "reject"}), config)
    state = graph.get_state(config).values
    assert state["status"] == "needs_human_review"
    assert state["operator_decision"] == "rejected"


def test_chat_memory_keeps_conversation_per_thread():
    seen = []

    def draft(messages, context, intent, model):
        seen.append(list(messages))
        return {"found": True, "answer": GOOD_REPLY}

    graph = build_graph(draft)
    config = solution.make_config("t-memory")
    graph.invoke(solution.initial_state("پیام اول", "test"), config)
    graph.invoke(solution.initial_state("پیام دوم", "test"), config)

    assert [m["role"] for m in seen[1]] == ["user", "assistant", "user"]
    state = graph.get_state(config).values
    assert len(state["messages"]) == 4
    assert state["messages"][-1] == {"role": "assistant", "content": GOOD_REPLY}


def test_model_error_does_not_crash_workflow():
    def broken(*args):
        raise ConnectionError("api down")

    graph = build_graph(broken, broken, broken)
    _, snapshot = run(graph, "t-error")
    state = snapshot.values
    assert state["error"] == "api down"
    assert state["score"] == 0
    assert state["revision_count"] == solution.MAX_REVISIONS
    assert state["status"] == "needs_human_review"


def test_cancel_order_waits_for_operator_then_updates_database():
    graph = build_graph(classify_fn=classifier("cancel_order", order_id=1))
    config, snapshot = run(graph, "t-cancel", "سفارش ۱ را لغو کنید", user=ALI)
    assert snapshot.interrupts[0].value["kind"] == "cancel_order"
    assert database.get_order(1)["status"] == "در حال پردازش"

    graph.invoke(Command(resume={"action": "approve"}), config)
    assert database.get_order(1)["status"] == "لغو شده"
    assert "لغو شد" in solution.last_assistant_message(graph.get_state(config).values)


def resume_customer(graph, config, text):
    graph.invoke(Command(resume={"text": text}), config)
    return graph.get_state(config)


def test_order_resolves_both_products_from_history_and_waits_for_customer():
    seen = []

    def extract(messages, catalog, current_cart, model):
        seen.append(list(messages))
        return cart((6, 1), (5, 1))

    stock = {pid: database.get_product(pid)["stock"] for pid in (5, 6)}
    orders_before = len(database.get_orders())
    graph = build_graph(answer("مشتری گرامی، در دسته‌بندی لوازم جانبی شارژر دیواری ۲۵ وات سامسونگ به قیمت ۹۵۰ هزار تومان و هدفون انکر به قیمت ۳ میلیون و ۴۰۰ هزار تومان موجود است."), classify_fn=classifier("question"), extract_fn=extract)
    config, _ = run(graph, "t-both", "یه سیم شارژر با یه هدفون میخوام، چی داری؟", user=ALI)

    solution.classify_message = classifier("place_order")
    graph.invoke(solution.initial_state("جفتشو برام ثبت کن", "test", ALI), config)
    snapshot = graph.get_state(config)

    assert [m["role"] for m in seen[0]] == ["user", "assistant", "user"]
    assert "هدفون انکر" in seen[0][1]["content"]
    pending = snapshot.interrupts[0].value
    assert (pending["kind"], pending["audience"]) == ("order_confirmation", "customer")
    assert "شارژر دیواری" in pending["customer_message"] and "هدفون بی‌سیم" in pending["customer_message"]
    assert "4,350,000" in pending["customer_message"]
    assert len(database.get_orders()) == orders_before

    state = resume_customer(graph, config, "بله").values
    order = database.get_order(database.get_orders()[-1]["id"])
    assert sorted((i["product_id"], i["quantity"]) for i in order["items"]) == [(5, 1), (6, 1)]
    assert order["total"] == 4350000
    assert database.get_product(5)["stock"] == stock[5] - 1
    assert database.get_product(6)["stock"] == stock[6] - 1
    assert state["cart"] is None
    assert f"شماره {order['id']}" in solution.last_assistant_message(state)


def test_customer_correction_rebuilds_cart_and_asks_again():
    graph = build_graph(
        classify_fn=classifier("place_order"),
        extract_fn=sequence(cart((6, 2)), cart((6, 1), (5, 1))),
        confirm_fn=sequence("modify", "confirm"),
    )
    config, snapshot = run(graph, "t-modify", "دو تا محصول رو میخوام", user=ALI)
    assert "2 عدد" in snapshot.interrupts[0].value["customer_message"]

    snapshot = resume_customer(graph, config, "نه، منظورم یه شارژر و یه هدفون بود")
    pending = snapshot.interrupts[0].value
    assert "هدفون بی‌سیم" in pending["customer_message"]
    assert snapshot.values["messages"][-1]["content"] == "نه، منظورم یه شارژر و یه هدفون بود"

    resume_customer(graph, config, "بله درسته")
    order = database.get_order(database.get_orders()[-1]["id"])
    assert sorted((i["product_id"], i["quantity"]) for i in order["items"]) == [(5, 1), (6, 1)]


def test_customer_can_cancel_order_before_saving():
    orders_before = len(database.get_orders())
    graph = build_graph(classify_fn=classifier("place_order"), extract_fn=lambda *a: cart((7, 1)),
                        confirm_fn=lambda *a: "cancel")
    config, _ = run(graph, "t-order-cancel-customer", "دوره پایتون میخوام", user=ALI)
    state = resume_customer(graph, config, "نه منصرف شدم").values
    assert len(database.get_orders()) == orders_before
    assert state["cart"] is None
    assert solution.last_assistant_message(state) == solution.ORDER_CANCELLED_MESSAGE


def test_order_uses_logged_in_customer_name_and_owner():
    graph = build_graph(classify_fn=classifier("place_order"), extract_fn=lambda *a: cart((6, 1)))
    config, snapshot = run(graph, "t-order-owner", "یه شارژر سامسونگ می‌خوام", user=MARYAM)
    assert "مریم احمدی" in snapshot.interrupts[0].value["customer_message"]

    graph.invoke(Command(resume={"text": "بله"}), config)
    order = database.get_order(database.get_orders()[-1]["id"])
    assert (order["user_id"], order["customer_name"]) == (MARYAM["id"], "مریم احمدی")


def test_account_actions_require_login():
    for intent in ("place_order", "order_status", "cancel_order", "return_order"):
        graph = build_graph(classify_fn=classifier(intent, order_id=1))
        _, snapshot = run(graph, f"t-login-{intent}", "سفارش ۱")
        assert not snapshot.interrupts
        assert solution.last_assistant_message(snapshot.values) == solution.LOGIN_REQUIRED_MESSAGE


def test_customer_cannot_see_or_change_other_customers_orders():
    status_before = database.get_order(1)["status"]
    for intent in ("order_status", "cancel_order", "return_order"):
        graph = build_graph(classify_fn=classifier(intent, order_id=1))
        _, snapshot = run(graph, f"t-other-{intent}", "سفارش ۱", user=MARYAM)
        reply = solution.last_assistant_message(snapshot.values)
        assert not snapshot.interrupts
        assert "در حساب کاربری شما پیدا نشد" in reply
        assert "علی محمدی" not in reply
        assert snapshot.values.get("active_order_id") is None
    assert database.get_order(1)["status"] == status_before


def test_authenticate_checks_password():
    assert database.authenticate("ali", "ali-1234")["full_name"] == "علی محمدی"
    assert database.authenticate("ali", "wrong") is None
    assert database.authenticate("nobody", "ali-1234") is None
    with database.get_connection() as conn:
        stored = conn.execute("SELECT password_hash FROM users WHERE username = 'ali'").fetchone()[0]
    assert "ali-1234" not in stored


def test_out_of_stock_product_is_not_ordered():
    graph = build_graph(classify_fn=classifier("place_order"), extract_fn=lambda *a: cart((2, 1)))
    _, snapshot = run(graph, "t-order-stock", "ردمی نوت ۱۳ می‌خوام", user=ALI)
    assert not snapshot.interrupts
    assert "ناموجود" in solution.last_assistant_message(snapshot.values)


def test_create_order_is_all_or_nothing():
    stock = database.get_product(6)["stock"]
    orders_before = len(database.get_orders())
    try:
        database.create_order(ALI["id"], "تست", [{"product_id": 6, "quantity": 1}, {"product_id": 2, "quantity": 1}])
        assert False, "expected OutOfStockError"
    except database.OutOfStockError as e:
        assert e.product_id == 2
    assert database.get_product(6)["stock"] == stock
    assert len(database.get_orders()) == orders_before


def test_cancel_restores_stock_of_every_item():
    order_id = database.create_order(ALI["id"], "نیما", [{"product_id": 3, "quantity": 1}, {"product_id": 6, "quantity": 2}])
    stock = {pid: database.get_product(pid)["stock"] for pid in (3, 6)}

    graph = build_graph(classify_fn=classifier("cancel_order", order_id=order_id))
    config, _ = run(graph, "t-order-cancel", f"سفارش {order_id} را لغو کنید", user=ALI)
    graph.invoke(Command(resume={"action": "approve"}), config)
    assert database.get_product(3)["stock"] == stock[3] + 1
    assert database.get_product(6)["stock"] == stock[6] + 2


def test_return_uses_order_from_conversation_and_waits_for_operator():
    with database.get_connection() as conn:
        conn.execute("UPDATE orders SET delivered_at = ? WHERE id = 3", (database.now(),))

    graph = build_graph(classify_fn=classifier("order_status", order_id=3))
    config, _ = run(graph, "t-return", "سفارش ۳ من کجاست؟", user=MOHAMMAD)

    solution.classify_message = classifier("return_order")
    graph.invoke(solution.initial_state("میخوام مرجوعش کنم", "test", MOHAMMAD), config)
    snapshot = graph.get_state(config)
    pending = snapshot.interrupts[0].value
    assert (pending["kind"], pending["audience"]) == ("return_order", "operator")
    assert "سفارش شماره 3" in pending["title"]
    assert database.get_order(3)["status"] == database.DELIVERED_STATUS

    graph.invoke(Command(resume={"action": "approve"}), config)
    assert database.get_order(3)["status"] == database.RETURN_APPROVED_STATUS
    assert "مرجوعی سفارش شماره 3" in solution.last_assistant_message(graph.get_state(config).values)

    graph.invoke(solution.initial_state("مرجوعش کن", "test", MOHAMMAD), config)
    snapshot = graph.get_state(config)
    assert not snapshot.interrupts
    assert "قبلاً تأیید شده" in solution.last_assistant_message(snapshot.values)


def test_return_outside_window_is_refused_without_operator():
    order_id = database.create_order(ALI["id"], "تست", [{"product_id": 7, "quantity": 1}])
    with database.get_connection() as conn:
        conn.execute("UPDATE orders SET status = ?, delivered_at = ? WHERE id = ?",
                     (database.DELIVERED_STATUS, "2026-01-01 10:00", order_id))

    graph = build_graph(classify_fn=classifier("return_order", order_id=order_id))
    _, snapshot = run(graph, "t-return-late", f"سفارش {order_id} را مرجوع کنید", user=ALI)
    assert not snapshot.interrupts
    assert "۷ روز" in solution.last_assistant_message(snapshot.values) or "7 روز" in solution.last_assistant_message(snapshot.values)


def test_return_of_unshipped_order_suggests_cancel():
    graph = build_graph(classify_fn=classifier("return_order", order_id=4))
    _, snapshot = run(graph, "t-return-unshipped", "سفارش ۴ را مرجوع کنید", user=ALI)
    assert not snapshot.interrupts
    assert "لغو کنید" in solution.last_assistant_message(snapshot.values)


def test_cancel_of_delivered_order_points_to_return():
    order_id = database.create_order(ALI["id"], "تست", [{"product_id": 7, "quantity": 1}])
    with database.get_connection() as conn:
        conn.execute("UPDATE orders SET status = ?, delivered_at = ? WHERE id = ?",
                     (database.DELIVERED_STATUS, database.now(), order_id))
    graph = build_graph(classify_fn=classifier("cancel_order", order_id=order_id))
    _, snapshot = run(graph, "t-cancel-delivered", f"سفارش {order_id} را لغو کنید", user=ALI)
    assert not snapshot.interrupts
    assert "مرجوع کنید" in solution.last_assistant_message(snapshot.values)


def test_cancel_order_rejected_keeps_order():
    graph = build_graph(classify_fn=classifier("cancel_order", order_id=4))
    config, _ = run(graph, "t-cancel-reject", "سفارش ۴ را لغو کنید", user=ALI)
    graph.invoke(Command(resume={"action": "reject"}), config)
    assert database.get_order(4)["status"] == "در انتظار پرداخت"


def test_shipped_order_cannot_be_cancelled_without_operator():
    graph = build_graph(classify_fn=classifier("cancel_order", order_id=2))
    _, snapshot = run(graph, "t-cancel-shipped", "سفارش ۲ را لغو کنید", user=MARYAM)
    assert not snapshot.interrupts
    assert database.get_order(2)["status"] == "ارسال شده"


def test_unanswered_question_offers_callback_and_saves_it_after_approval():
    graph = build_graph(draft_fn=NOT_FOUND, classify_fn=classifier("question"))
    config, snapshot = run(graph, "t-callback", "آیا شعبه حضوری در شیراز دارید؟")
    state = snapshot.values
    assert state["awaiting_callback"]
    assert solution.last_assistant_message(state) == solution.CALLBACK_OFFER_MESSAGE

    solution.classify_message = classifier("confirm_yes", phone="09121234567")
    graph.invoke(solution.initial_state("بله، ۰۹۱۲۱۲۳۴۵۶۷", "test"), config)
    assert graph.get_state(config).interrupts[0].value["kind"] == "callback"
    assert not database.get_callback_requests()

    graph.invoke(Command(resume={"action": "approve"}), config)
    requests = database.get_callback_requests()
    assert requests[0]["phone"] == "09121234567"
    assert requests[0]["question"] == "آیا شعبه حضوری در شیراز دارید؟"
    assert not graph.get_state(config).values["awaiting_callback"]


def test_callback_no_answer_closes_offer():
    graph = build_graph(draft_fn=NOT_FOUND, classify_fn=classifier("question"))
    config, _ = run(graph, "t-callback-no", "شعبه حضوری دارید؟")
    solution.classify_message = classifier("confirm_no")
    graph.invoke(solution.initial_state("نه ممنون", "test"), config)
    state = graph.get_state(config).values
    assert not state["awaiting_callback"]
    assert solution.last_assistant_message(state) == solution.CALLBACK_DECLINED_MESSAGE


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"✅ {name}")
