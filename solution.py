import argparse
import operator
import os
import sys
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig

from critic import score_reply, PASS_SCORE
from database import (
    authenticate, get_order, cancel_order, approve_return, create_callback_request, get_products, create_order,
    CANCELLABLE_STATUSES, NEW_ORDER_STATUS, DELIVERED_STATUS, RETURN_APPROVED_STATUS, RETURN_WINDOW_DAYS,
    TIME_FORMAT, OutOfStockError
)
from rag import retrieve, reset_index
from services import (
    generate_draft, revise_draft, classify_message, extract_order, interpret_confirmation, DEFAULT_MODEL
)

sys.stdout.reconfigure(encoding="utf-8")

MAX_REVISIONS = 2
RECURSION_LIMIT = 15
HISTORY_WINDOW = 12
ORDER_INTENTS = ("cancel_order", "return_order", "order_status")
ACCOUNT_INTENTS = ORDER_INTENTS + ("place_order",)
DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@localhost:5432/postgres"
DEFAULT_COMPLAINT = "سه روز است مبلغ از حسابم کم شده اما سفارش ثبت نشده"

REVIEW_WAITING_MESSAGE = """🧑‍💼 درخواست شما برای بررسی دقیق‌تر به کارشناس پشتیبانی ارجاع شد.
لطفاً چند لحظه منتظر بمانید؛ پاسخ در همین صفحه نمایش داده می‌شود."""
REJECTED_MESSAGE = "مشتری گرامی، درخواست شما برای پیگیری دقیق‌تر به کارشناس پشتیبانی ارجاع شد و کارشناس ما به‌زودی با شما تماس می‌گیرد."
ASK_ORDER_ID_MESSAGE = "مشتری گرامی، لطفاً شماره سفارش خود را بنویسید تا آن را بررسی کنیم."
CALLBACK_OFFER_MESSAGE = """مشتری گرامی، متأسفانه پاسخ سؤال شما را در اطلاعات پشتیبانی پیدا نکردیم.
آیا مایلید کارشناسان ما تا ۴۸ ساعت آینده با شما تماس بگیرند؟ در صورت تمایل «بله» را همراه با شماره تماس خود بنویسید."""
CALLBACK_DECLINED_MESSAGE = "مشتری گرامی، متوجه شدیم. اگر سؤال دیگری داشتید، در خدمت شما هستیم."
LOGIN_REQUIRED_MESSAGE = "مشتری گرامی، برای مشاهده، ثبت، لغو یا مرجوعی سفارش لطفاً ابتدا وارد حساب کاربری خود شوید."
ORDER_CANCELLED_MESSAGE = "مشتری گرامی، سفارش ثبت نشد. هر زمان خواستید می‌توانید دوباره سفارش دهید."


class ComplaintState(TypedDict):
    complaint: str
    model: str
    intent: str
    user_id: Optional[int]
    user_name: Optional[str]
    order_id: Optional[int]
    active_order_id: Optional[int]
    phone: Optional[str]
    cart: Optional[dict]
    order_ready: bool
    order_decision: str
    context: List[str]
    draft: str
    not_found: bool
    score: int
    feedback: List[str]
    revision_count: int
    status: str
    operator_decision: str
    error: str
    awaiting_callback: bool
    pending_question: str
    messages: Annotated[List[dict], operator.add]
    history: Annotated[List[str], operator.add]


def assistant_message(text):
    return {"role": "assistant", "content": text}


def reply(text, step, **updates):
    return {
        "status": "approved",
        "messages": [assistant_message(text)],
        "history": [step],
        **updates
    }


def classify_node(state: ComplaintState) -> dict:
    awaiting_callback = state.get("awaiting_callback", False)
    try:
        result = classify_message(
            state["complaint"], awaiting_callback, bool(state.get("cart")), state.get("active_order_id"),
            state["messages"][-HISTORY_WINDOW:-1], state["model"]
        )
        error = ""
    except Exception as e:
        print(f"⚠️ خطا در تشخیص نیت کاربر: {e}")
        result = {"intent": "complaint", "order_id": None, "phone": None}
        error = str(e)

    if result["intent"] in ORDER_INTENTS and not result["order_id"]:
        result["order_id"] = state.get("active_order_id")

    return {
        **result,
        "awaiting_callback": awaiting_callback and result["intent"] in ("confirm_yes", "confirm_no"),
        "error": error,
        "history": [f"classify: {result['intent']}"]
    }


def retrieve_node(state: ComplaintState) -> dict:
    intent = state["intent"] if state["intent"] in ("question", "complaint") else "question"
    try:
        context = retrieve(state["complaint"])
    except Exception as e:
        print(f"⚠️ خطا در جستجوی پایگاه دانش: {e}")
        context = []

    return {
        "intent": intent,
        "context": context,
        "history": ["retrieve"]
    }


def draft_node(state: ComplaintState) -> dict:
    try:
        result = generate_draft(state["messages"], state["context"], state["intent"], state["model"])
        error = ""
    except Exception as e:
        print(f"⚠️ خطا در تولید پیش‌نویس: {e}")
        result = {"found": True, "answer": ""}
        error = str(e)

    not_found = not result["found"]
    return {
        "draft": result["answer"],
        "not_found": not_found,
        "error": error,
        "history": ["draft: not found" if not_found else "draft"]
    }


def critique_node(state: ComplaintState) -> dict:
    score, feedback = score_reply(state["draft"], state["intent"])
    return {
        "score": score,
        "feedback": feedback,
        "history": [f"critique: {score}"]
    }


def revise_node(state: ComplaintState) -> dict:
    revision_count = state["revision_count"] + 1
    try:
        draft = revise_draft(
            state["complaint"], state["draft"], state["feedback"],
            state["context"], state["intent"], state["model"]
        )
        error = ""
    except Exception as e:
        print(f"⚠️ خطا در بازنویسی پاسخ: {e}")
        draft = state["draft"]
        error = str(e)

    return {
        "draft": draft,
        "revision_count": revision_count,
        "error": error,
        "history": [f"revise #{revision_count}"]
    }


def finalize_node(state: ComplaintState) -> dict:
    if state["score"] >= PASS_SCORE:
        return reply(state["draft"], "finalize: approved")
    return {
        "status": "needs_human_review",
        "history": ["finalize: needs_human_review"]
    }


def human_review_node(state: ComplaintState) -> dict:
    feedback = "\n".join(f"- {item}" for item in state["feedback"])
    decision = interrupt({
        "kind": "reply_review",
        "audience": "operator",
        "title": "✍️ بازبینی پاسخ مدل",
        "summary": f"📊 امتیاز پیش‌نویس: **{state['score']}/100** (پس از {MAX_REVISIONS} بار بازنویسی)\n\n⚠️ **ایرادهای منتقد:**\n{feedback}",
        "draft": state["draft"],
        "customer_message": REVIEW_WAITING_MESSAGE,
    })

    action = decision.get("action")
    if action == "approve":
        return reply(state["draft"], "operator: approved", operator_decision="approved")
    if action == "edit":
        score, feedback = score_reply(decision["text"], state["intent"])
        return reply(
            decision["text"], "operator: edited",
            draft=decision["text"], score=score, feedback=feedback, operator_decision="edited"
        )
    return {
        "operator_decision": "rejected",
        "messages": [assistant_message(REJECTED_MESSAGE)],
        "history": ["operator: rejected"]
    }


def offer_callback_node(state: ComplaintState) -> dict:
    return reply(
        CALLBACK_OFFER_MESSAGE, "offer_callback",
        awaiting_callback=True, pending_question=state["complaint"]
    )


def load_customer_order(state, step):
    order_id = state["order_id"]
    if not order_id:
        return None, reply(ASK_ORDER_ID_MESSAGE, f"{step}: no id")

    order = get_order(order_id)
    if not order or order["user_id"] != state.get("user_id"):
        return None, reply(
            f"مشتری گرامی، سفارشی با شماره {order_id} در حساب کاربری شما پیدا نشد. لطفاً شماره سفارش را بررسی کنید.",
            f"{step}: not found"
        )
    return order, None


def login_required_node(state: ComplaintState) -> dict:
    return reply(LOGIN_REQUIRED_MESSAGE, "login_required")


def order_status_node(state: ComplaintState) -> dict:
    order, early_reply = load_customer_order(state, "order_status")
    if early_reply:
        return early_reply
    order_id = order["id"]

    return reply(
        f"مشتری گرامی، سفارش شماره {order_id} در وضعیت «{order['status']}» قرار دارد.\n\n{format_items(order['items'])}",
        "order_status", active_order_id=order_id
    )


def cancel_order_node(state: ComplaintState) -> dict:
    order, early_reply = load_customer_order(state, "cancel_order")
    if early_reply:
        return early_reply
    order_id = order["id"]

    if order["status"] not in CANCELLABLE_STATUSES:
        hint = (
            f" اگر می‌خواهید کالا را مرجوع کنید، کافی است بنویسید «سفارش {order_id} را مرجوع کنید»."
            if order["status"] == DELIVERED_STATUS else ""
        )
        return reply(
            f"مشتری گرامی، سفارش شماره {order_id} در وضعیت «{order['status']}» است و امکان لغو آن وجود ندارد.{hint}",
            "cancel_order: not cancellable", active_order_id=order_id
        )

    decision = interrupt({
        "kind": "cancel_order",
        "audience": "operator",
        "title": f"🛑 درخواست لغو سفارش شماره {order_id}",
        "summary": f"""👤 مشتری: {order['customer_name']}
📌 وضعیت فعلی: {order['status']}
📅 تاریخ ثبت: {order['created_at']}

{format_items(order['items'])}""",
        "customer_message": f"🧑‍💼 درخواست لغو سفارش شماره {order_id} ثبت شد و پس از تأیید کارشناس انجام می‌شود. لطفاً چند لحظه منتظر بمانید.",
    })

    if decision.get("action") == "approve":
        cancel_order(order_id)
        reset_index()
        return reply(
            f"مشتری گرامی، سفارش شماره {order_id} با تأیید کارشناس لغو شد. "
            "در صورت پرداخت، مبلغ طبق روال بانکی طی ۷۲ ساعت کاری به حساب شما برمی‌گردد.",
            "cancel_order: approved", operator_decision="approved", active_order_id=order_id
        )
    return reply(
        f"مشتری گرامی، درخواست لغو سفارش شماره {order_id} توسط کارشناس تأیید نشد. "
        "برای پیگیری بیشتر می‌توانید با پشتیبانی تلفنی تماس بگیرید.",
        "cancel_order: rejected", operator_decision="rejected", active_order_id=order_id
    )


def days_since(timestamp):
    return (datetime.now() - datetime.strptime(timestamp, TIME_FORMAT)).days


def return_order_node(state: ComplaintState) -> dict:
    order, early_reply = load_customer_order(state, "return_order")
    if early_reply:
        return early_reply
    order_id = order["id"]

    if order["status"] in CANCELLABLE_STATUSES:
        return reply(
            f"مشتری گرامی، سفارش شماره {order_id} هنوز ارسال نشده و نیازی به مرجوعی ندارد؛ "
            f"اگر آن را نمی‌خواهید، کافی است بنویسید «سفارش {order_id} را لغو کنید».",
            "return_order: not shipped", active_order_id=order_id
        )

    if order["status"] == RETURN_APPROVED_STATUS:
        return reply(
            f"مشتری گرامی، درخواست مرجوعی سفارش شماره {order_id} قبلاً تأیید شده است و کارشناسان ما برای هماهنگی تحویل کالا با شما تماس می‌گیرند.",
            "return_order: already approved", active_order_id=order_id
        )

    if order["status"] != DELIVERED_STATUS or not order["delivered_at"]:
        return reply(
            f"مشتری گرامی، سفارش شماره {order_id} در وضعیت «{order['status']}» است و فقط سفارش‌های تحویل‌داده‌شده قابل مرجوعی هستند.",
            "return_order: not delivered", active_order_id=order_id
        )

    days = days_since(order["delivered_at"])
    if days > RETURN_WINDOW_DAYS:
        return reply(
            f"مشتری گرامی، سفارش شماره {order_id} در تاریخ {order['delivered_at']} تحویل داده شده و {days} روز از آن گذشته است. "
            f"طبق شرایط فروشگاه، مرجوعی فقط تا {RETURN_WINDOW_DAYS} روز پس از تحویل امکان‌پذیر است.",
            "return_order: window passed", active_order_id=order_id
        )

    decision = interrupt({
        "kind": "return_order",
        "audience": "operator",
        "title": f"↩️ درخواست مرجوعی سفارش شماره {order_id}",
        "summary": f"""👤 مشتری: {order['customer_name']}
📦 تاریخ تحویل: {order['delivered_at']} ({days} روز پیش؛ مهلت مرجوعی {RETURN_WINDOW_DAYS} روز)
💬 پیام مشتری: {state['complaint']}

{format_items(order['items'])}

⚠️ پیش از تأیید، سالم بودن کالا و بسته‌بندی را طبق شرایط مرجوعی بررسی کنید.""",
        "customer_message": f"🧑‍💼 درخواست مرجوعی سفارش شماره {order_id} ثبت شد و پس از تأیید کارشناس انجام می‌شود. لطفاً چند لحظه منتظر بمانید.",
    })

    if decision.get("action") == "approve":
        approve_return(order_id)
        return reply(
            f"مشتری گرامی، درخواست مرجوعی سفارش شماره {order_id} با تأیید کارشناس ثبت شد. "
            "لطفاً کالا را با بسته‌بندی سالم آماده کنید؛ کارشناسان ما برای هماهنگی تحویل کالا با شما تماس می‌گیرند.",
            "return_order: approved", operator_decision="approved", active_order_id=order_id
        )
    return reply(
        f"مشتری گرامی، درخواست مرجوعی سفارش شماره {order_id} توسط کارشناس تأیید نشد. "
        "برای پیگیری بیشتر می‌توانید با پشتیبانی تلفنی تماس بگیرید.",
        "return_order: rejected", operator_decision="rejected", active_order_id=order_id
    )


def format_items(rows):
    lines = [f"• {row['product_name']} — {row['quantity']} عدد × {row['unit_price']:,} تومان" for row in rows]
    total = sum(row["quantity"] * row["unit_price"] for row in rows)
    return "\n".join(lines) + f"\n\nمبلغ کل: {total:,} تومان"


def cart_rows(cart, products):
    return [
        {"product_name": products[item["product_id"]]["name"], "quantity": item["quantity"],
         "unit_price": products[item["product_id"]]["price"]}
        for item in cart["items"]
    ]


def merge_items(items, products):
    merged = {}
    for item in items:
        if item["product_id"] in products and item["quantity"] > 0:
            merged[item["product_id"]] = merged.get(item["product_id"], 0) + item["quantity"]
    return [{"product_id": product_id, "quantity": quantity} for product_id, quantity in merged.items()]


def build_order_node(state: ComplaintState) -> dict:
    cart = state.get("cart") or {"items": []}
    products = {p["id"]: p for p in get_products()}
    catalog = "\n".join(f"{p['id']}: {p['name']} — {p['price']:,}" for p in products.values())

    try:
        extracted = extract_order(state["messages"][-HISTORY_WINDOW:], catalog, cart, state["model"])
        error = ""
    except Exception as e:
        print(f"⚠️ خطا در استخراج سبد خرید: {e}")
        extracted = cart
        error = str(e)

    cart = {"items": merge_items(extracted["items"], products)}

    if not cart["items"]:
        available = "\n".join(f"• {p['name']} — {p['price']:,} تومان" for p in products.values() if p["stock"] > 0)
        return reply(
            f"مشتری گرامی، لطفاً بنویسید کدام محصولات و از هرکدام چند عدد می‌خواهید.\n\nمحصولات موجود:\n{available}",
            "build_order: no items", cart=cart, order_ready=False, error=error
        )

    shortages = [
        f"• {products[item['product_id']]['name']}: "
        + ("ناموجود" if products[item["product_id"]]["stock"] == 0 else f"فقط {products[item['product_id']]['stock']} عدد موجود است")
        for item in cart["items"] if products[item["product_id"]]["stock"] < item["quantity"]
    ]
    if shortages:
        return reply(
            "مشتری گرامی، موجودی این محصولات کافی نیست:\n" + "\n".join(shortages)
            + "\n\nلطفاً تعداد را کمتر کنید یا محصول دیگری انتخاب کنید.",
            "build_order: out of stock", cart=cart, order_ready=False, error=error
        )

    return {
        "cart": cart,
        "order_ready": True,
        "error": error,
        "history": ["build_order: ready"]
    }


def confirm_order_node(state: ComplaintState) -> dict:
    cart = state["cart"]
    products = {p["id"]: p for p in get_products()}
    summary = f"{format_items(cart_rows(cart, products))}\nنام سفارش‌دهنده: {state['user_name']}"

    answer = interrupt({
        "kind": "order_confirmation",
        "audience": "customer",
        "summary": summary,
        "customer_message": f"""مشتری گرامی، لطفاً سفارش خود را بررسی کنید:

{summary}

آیا سفارش ثبت شود؟ «بله» یا «خیر» را بنویسید؛ اگر موردی اشتباه است، اصلاح آن را بنویسید.""",
    })

    text = answer["text"]
    try:
        decision = interpret_confirmation(summary, text, state["model"])
    except Exception as e:
        print(f"⚠️ خطا در تفسیر پاسخ مشتری: {e}")
        decision = "modify"

    update = {
        "complaint": text,
        "order_decision": decision,
        "messages": [{"role": "user", "content": text}],
        "history": [f"confirm_order: {decision}"]
    }
    if decision == "cancel":
        update["messages"].append(assistant_message(ORDER_CANCELLED_MESSAGE))
        update["cart"] = None
        update["status"] = "approved"
    return update


def save_order_node(state: ComplaintState) -> dict:
    cart = state["cart"]
    try:
        order_id = create_order(state["user_id"], state["user_name"], cart["items"])
    except OutOfStockError as e:
        product = {p["id"]: p for p in get_products()}[e.product_id]
        return reply(
            f"مشتری گرامی، متأسفانه موجودی {product['name']} در همین فاصله کافی نیست و سفارش ثبت نشد. "
            "لطفاً سبد خود را اصلاح کنید.",
            "save_order: out of stock"
        )

    reset_index()
    order = get_order(order_id)
    return reply(
        f"""مشتری گرامی، سفارش شما با شماره {order_id} ثبت شد.

{format_items(order['items'])}
وضعیت: {NEW_ORDER_STATUS}

برای پیگیری یا لغو سفارش، کافی است شماره سفارش را در همین چت بنویسید.""",
        "save_order: created", cart=None, active_order_id=order_id
    )


def request_callback_node(state: ComplaintState, config: RunnableConfig) -> dict:
    phone = state.get("phone")
    question = state.get("pending_question", "")

    decision = interrupt({
        "kind": "callback",
        "audience": "operator",
        "title": "📞 درخواست تماس تلفنی",
        "summary": f"❓ **سؤال بی‌پاسخ:** {question}\n\n📱 **شماره تماس:** {phone or 'ثبت نشده'}",
        "customer_message": "🧑‍💼 درخواست تماس شما برای تأیید به کارشناس ارسال شد. لطفاً چند لحظه منتظر بمانید.",
    })

    if decision.get("action") == "approve":
        request_id = create_callback_request(config["configurable"]["thread_id"], phone, question)
        return reply(
            f"مشتری گرامی، درخواست تماس شما با کد پیگیری {request_id} ثبت شد و کارشناسان ما تا ۴۸ ساعت آینده با شما تماس می‌گیرند.",
            "callback: approved", operator_decision="approved", awaiting_callback=False
        )
    return reply(
        "مشتری گرامی، امکان ثبت درخواست تماس در حال حاضر وجود ندارد. لطفاً در ساعات کاری با پشتیبانی تلفنی تماس بگیرید.",
        "callback: rejected", operator_decision="rejected", awaiting_callback=False
    )


def decline_callback_node(state: ComplaintState) -> dict:
    return reply(CALLBACK_DECLINED_MESSAGE, "callback: declined", awaiting_callback=False)


def route_after_classify(state: ComplaintState) -> str:
    intent = state["intent"]
    if intent in ACCOUNT_INTENTS and not state.get("user_id"):
        return "login_required"
    if intent == "place_order":
        return "build_order"
    if intent == "cancel_order":
        return "cancel_order"
    if intent == "return_order":
        return "return_order"
    if intent == "order_status":
        return "order_status"
    if state["awaiting_callback"] and intent == "confirm_yes":
        return "request_callback"
    if state["awaiting_callback"] and intent == "confirm_no":
        return "decline_callback"
    return "retrieve"


def route_after_build_order(state: ComplaintState) -> str:
    return "confirm_order" if state["order_ready"] else END


def route_after_confirm_order(state: ComplaintState) -> str:
    return {"confirm": "save_order", "modify": "build_order"}.get(state["order_decision"], END)


def route_after_draft(state: ComplaintState) -> str:
    return "offer_callback" if state["not_found"] else "critique"


def route_after_critique(state: ComplaintState) -> str:
    if state["score"] >= PASS_SCORE or state["revision_count"] >= MAX_REVISIONS:
        return "finalize"
    return "revise"


def route_after_finalize(state: ComplaintState) -> str:
    if state["status"] == "needs_human_review":
        return "human_review"
    return END


def create_complaint_graph(checkpointer):
    workflow = StateGraph(ComplaintState)

    workflow.add_node("classify", classify_node)
    workflow.add_node("login_required", login_required_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("offer_callback", offer_callback_node)
    workflow.add_node("order_status", order_status_node)
    workflow.add_node("build_order", build_order_node)
    workflow.add_node("confirm_order", confirm_order_node)
    workflow.add_node("save_order", save_order_node)
    workflow.add_node("cancel_order", cancel_order_node)
    workflow.add_node("return_order", return_order_node)
    workflow.add_node("request_callback", request_callback_node)
    workflow.add_node("decline_callback", decline_callback_node)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify", route_after_classify,
        ["login_required", "build_order", "cancel_order", "return_order", "order_status", "request_callback", "decline_callback", "retrieve"]
    )
    workflow.add_conditional_edges("build_order", route_after_build_order, ["confirm_order", END])
    workflow.add_conditional_edges("confirm_order", route_after_confirm_order, ["save_order", "build_order", END])
    workflow.add_edge("retrieve", "draft")
    workflow.add_conditional_edges("draft", route_after_draft, ["offer_callback", "critique"])
    workflow.add_conditional_edges("critique", route_after_critique, ["finalize", "revise"])
    workflow.add_edge("revise", "critique")
    workflow.add_conditional_edges("finalize", route_after_finalize, ["human_review", END])

    for node in ["login_required", "human_review", "offer_callback", "order_status", "save_order", "cancel_order", "return_order", "request_callback", "decline_callback"]:
        workflow.add_edge(node, END)

    return workflow.compile(checkpointer=checkpointer)


def new_thread_id():
    return f"customer-{uuid.uuid4().hex[:6]}"


def make_config(thread_id):
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT
    }


def initial_state(complaint, model, user=None):
    return {
        "complaint": complaint,
        "user_id": user["id"] if user else None,
        "user_name": user["full_name"] if user else None,
        "model": model,
        "intent": "",
        "order_id": None,
        "phone": None,
        "order_ready": False,
        "order_decision": "",
        "context": [],
        "draft": "",
        "not_found": False,
        "score": 0,
        "feedback": [],
        "revision_count": 0,
        "status": "in_review",
        "operator_decision": "",
        "error": "",
        "messages": [{"role": "user", "content": complaint}],
        "history": []
    }


def run_workflow(graph, config, payload):
    try:
        graph.invoke(payload, config)
    except GraphRecursionError:
        print("⚠️ سقف recursion_limit رد شد؛ پاسخ به بازبینی انسانی ارجاع می‌شود.")
        graph.update_state(config, {"status": "needs_human_review"})


def last_assistant_message(state):
    for message in reversed(state.get("messages", [])):
        if message["role"] == "assistant":
            return message["content"]
    return ""


def ask_operator(review):
    if review.get("audience") == "customer":
        print(f"\n{review['customer_message']}\n")
        try:
            text = input("پاسخ مشتری (Enter=بعداً): ").strip()
        except EOFError:
            text = ""
        return {"text": text} if text else None

    print(f"""
🧑‍💼 {review['title']}
{'═' * 40}
{review['summary']}
""")
    if review.get("draft") is not None:
        print(f"💬 پیش‌نویس:\n{review['draft'] or '(خالی)'}\n")

    options = "a=تأیید / e=ویرایش / r=رد" if review["kind"] == "reply_review" else "a=تأیید / r=رد"
    try:
        choice = input(f"تصمیم اپراتور [{options} / Enter=بعداً]: ").strip().lower()
        if choice == "a":
            return {"action": "approve"}
        if choice == "e" and review["kind"] == "reply_review":
            return {"action": "edit", "text": input("متن نهایی: ").strip()}
        if choice == "r":
            return {"action": "reject"}
    except EOFError:
        pass
    return None


def handle_operator(graph, config):
    snapshot = graph.get_state(config)
    if not snapshot.interrupts:
        return

    decision = ask_operator(snapshot.interrupts[0].value)
    if decision is None:
        thread_id = config["configurable"]["thread_id"]
        print(f"\n⏸️ بازبینی در PostgreSQL ذخیره شد. برای ادامه:\npython solution.py --thread-id {thread_id}")
        return

    run_workflow(graph, config, Command(resume=decision))


def format_report(graph, config):
    snapshot = graph.get_state(config)
    state = snapshot.values
    checkpoints = len(list(graph.get_state_history(config)))
    status_icon = "✅" if state["status"] == "approved" else "🧑‍💼"
    final_reply = snapshot.interrupts[0].value["customer_message"] if snapshot.interrupts else last_assistant_message(state)

    report = f"""
📋 **نتیجهٔ پردازش پیام**

🧵 thread_id: `{config['configurable']['thread_id']}`

📝 **پیام مشتری:** {state['complaint']}
🎯 **نیت تشخیص داده‌شده:** {state['intent']}

💬 **پاسخ به مشتری:**

{final_reply or '(خالی)'}

---
📊 امتیاز: **{state['score']}/100**
🔁 تعداد بازنویسی: **{state['revision_count']}**
{status_icon} وضعیت: **{state['status']}**
"""
    if state["operator_decision"]:
        report += f"🧑‍💼 تصمیم اپراتور: **{state['operator_decision']}**  \n"
    if snapshot.interrupts:
        report += "⏸️ منتظر تصمیم اپراتور  \n"
    if state["feedback"]:
        report += "\n⚠️ **ایرادهای منتقد:**\n\n" + "\n".join(f"- {item}" for item in state["feedback"]) + "\n"
    if state["error"]:
        report += f"\n❌ آخرین خطای مدل: {state['error']}\n"

    report += f"\n📜 مسیر اجرا: {' → '.join(state['history'])}  "
    report += f"\n💾 تعداد checkpointها: {checkpoints}\n"
    return report


def open_postgres_checkpointer():
    from psycopg import Connection
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver

    conn = Connection.connect(
        os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer


def parse_args():
    parser = argparse.ArgumentParser(description="Customer support chatbot workflow")
    parser.add_argument("--complaint", help="پیام مشتری")
    parser.add_argument("--thread-id", help="شناسهٔ thread برای ذخیره و ادامهٔ state")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--username", help="نام کاربری برای کارهای مربوط به سفارش")
    parser.add_argument("--password", help="رمز عبور")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("AVALAI_API_KEY"):
        print("⚠️ AVALAI_API_KEY در فایل .env خالی است؛ پاسخ‌ها به بازبینی انسانی می‌روند.")

    user = None
    if args.username:
        user = authenticate(args.username, args.password or "")
        if not user:
            print("❌ نام کاربری یا رمز عبور اشتباه است.")
            return

    graph = create_complaint_graph(open_postgres_checkpointer())

    thread_id = args.thread_id or new_thread_id()
    config = make_config(thread_id)
    has_state = bool(graph.get_state(config).values)

    if args.complaint or not has_state:
        run_workflow(graph, config, initial_state(args.complaint or DEFAULT_COMPLAINT, args.model, user))

    handle_operator(graph, config)
    print(format_report(graph, config))


if __name__ == "__main__":
    main()
