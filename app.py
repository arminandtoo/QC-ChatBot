import gradio as gr
from langgraph.types import Command

from database import authenticate, get_user, get_orders, get_callback_requests
from services import DEFAULT_MODEL
from solution import (
    create_complaint_graph, open_postgres_checkpointer, make_config, initial_state,
    run_workflow, new_thread_id, last_assistant_message, MAX_REVISIONS
)

WELCOME_MESSAGE = """سلام {name}! 👋 به پشتیبانی آنلاین خوش آمدید.

می‌توانید درباره محصولات، قیمت و موجودی، ارسال، مرجوعی و وضعیت سفارش خود سؤال کنید،
درخواست لغو سفارش بدهید یا مشکل خود را با ما در میان بگذارید."""

ACCESS_DENIED_MESSAGE = "⛔ این صفحه فقط برای اپراتور پشتیبانی است."

STILL_WAITING_MESSAGE = "⏳ کارشناس ما در حال بررسی پیام قبلی شماست. لطفاً تا دریافت پاسخ منتظر بمانید."

KIND_LABELS = {
    "reply_review": "✍️ بازبینی پاسخ",
    "cancel_order": "🛑 لغو سفارش",
    "return_order": "↩️ مرجوعی",
    "callback": "📞 درخواست تماس",
}

_checkpointer = None
_graph = None


def get_graph():
    global _checkpointer, _graph
    if _graph is None:
        _checkpointer = open_postgres_checkpointer()
        _graph = create_complaint_graph(_checkpointer)
    return _graph


def pending_interrupt(snapshot):
    if not snapshot.interrupts:
        return None
    return {"audience": "operator", **snapshot.interrupts[0].value}


def get_status_text(state, pending):
    if pending and pending["audience"] == "customer":
        return "🛒 منتظر تأیید سفارش توسط شما"
    if pending:
        return "🧑‍💼 در انتظار تأیید کارشناس"
    if state["history"][-1].startswith(("finalize", "operator")):
        return f"✅ پاسخ پس از بازبینی کیفیت ارسال شد | 📊 امتیاز: {state['score']}/100 | 🔁 بازنویسی: {state['revision_count']}/{MAX_REVISIONS}"
    return "✅ پاسخ ارسال شد"


def current_user(request):
    return get_user(request.username) if request and request.username else None


def is_operator(request):
    user = current_user(request)
    return bool(user) and user["role"] == "operator"


def start_chat(request: gr.Request):
    user = current_user(request)
    welcome = WELCOME_MESSAGE.format(name=user["full_name"] if user else "")
    status = f"👤 {user['full_name']} | 💬 آماده دریافت پیام شما" if user else "💬 آماده دریافت پیام شما"
    return [{"role": "assistant", "content": welcome}], new_thread_id(), status, False


def send_message(message, history, thread_id, waiting, request: gr.Request):
    history = history or []

    if not message or not message.strip():
        return history, "", gr.skip(), waiting

    history.append({"role": "user", "content": message.strip()})

    if waiting:
        history.append({"role": "assistant", "content": STILL_WAITING_MESSAGE})
        return history, "", gr.skip(), waiting

    graph = get_graph()
    config = make_config(thread_id)

    pending = pending_interrupt(graph.get_state(config))
    if pending and pending["audience"] == "customer":
        run_workflow(graph, config, Command(resume={"text": message.strip()}))
    else:
        run_workflow(graph, config, initial_state(message.strip(), DEFAULT_MODEL, current_user(request)))

    snapshot = graph.get_state(config)
    pending = pending_interrupt(snapshot)
    reply = pending["customer_message"] if pending else last_assistant_message(snapshot.values)
    waiting_for_operator = bool(pending) and pending["audience"] == "operator"

    history.append({"role": "assistant", "content": reply})
    return history, "", get_status_text(snapshot.values, pending), waiting_for_operator


def check_operator_reply(history, thread_id, waiting):
    if not waiting:
        return gr.skip(), gr.skip(), gr.skip()

    snapshot = get_graph().get_state(make_config(thread_id))
    if snapshot.interrupts:
        return gr.skip(), gr.skip(), gr.skip()

    history = history + [{"role": "assistant", "content": last_assistant_message(snapshot.values)}]
    return history, "✅ پاسخ کارشناس ارسال شد", False


def list_pending_choices():
    graph = get_graph()
    # NOTE: scans every thread in Postgres, keep a pending-reviews table if threads grow large
    rows = _checkpointer.conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()

    choices = []
    for row in sorted(rows, key=lambda r: r["thread_id"]):
        pending = pending_interrupt(graph.get_state(make_config(row["thread_id"])))
        if pending and pending["audience"] == "operator":
            choices.append((f"{KIND_LABELS.get(pending['kind'], pending['kind'])} | {row['thread_id']}", row["thread_id"]))
    return choices


def show_review(thread_id, request: gr.Request):
    hidden = gr.update(value="", visible=False)
    if not is_operator(request):
        return ACCESS_DENIED_MESSAGE, hidden, gr.update(visible=False)
    if not thread_id:
        return "✅ موردی برای بازبینی وجود ندارد.", hidden, gr.update(visible=False)

    snapshot = get_graph().get_state(make_config(thread_id))
    review = pending_interrupt(snapshot)
    if not review or review["audience"] != "operator":
        return "✅ این مورد قبلاً بررسی شده است.", hidden, gr.update(visible=False)

    conversation = "\n\n".join(
        f"**{'👤 مشتری' if m['role'] == 'user' else '🤖 پشتیبانی'}:** {m['content']}"
        for m in snapshot.values["messages"]
    )

    text = f"""
## {review['title']}

{review['summary']}

---
### 💬 گفتگو
{conversation}
"""
    is_reply = review["kind"] == "reply_review"
    return text, gr.update(value=review.get("draft", ""), visible=is_reply), gr.update(visible=is_reply)


def get_tables():
    orders = [
        [o["id"], o["owner"], o["customer_name"], o["items"], f"{o['total']:,}", o["status"], o["updated_at"] or "-"]
        for o in get_orders()
    ]
    callbacks = [
        [c["id"], c["thread_id"], c["phone"] or "-", c["question"], c["status"], c["created_at"]]
        for c in get_callback_requests()
    ]
    return orders, callbacks


def refresh_operator(request: gr.Request):
    if not is_operator(request):
        hidden = gr.update(value="", visible=False)
        return (
            gr.update(choices=[], value=None), ACCESS_DENIED_MESSAGE, hidden, gr.update(visible=False),
            ACCESS_DENIED_MESSAGE, [], []
        )

    choices = list_pending_choices()
    selected = choices[0][1] if choices else None
    review, draft, edit_btn = show_review(selected, request)
    orders, callbacks = get_tables()
    return (
        gr.update(choices=choices, value=selected), review, draft, edit_btn,
        f"📥 موارد در انتظار: {len(choices)}", orders, callbacks
    )


def operator_decide(action, thread_id, edited_text, request):
    if not is_operator(request):
        return refresh_operator(request)

    graph = get_graph()
    config = make_config(thread_id or "")

    pending = pending_interrupt(graph.get_state(config)) if thread_id else None
    if pending and pending["audience"] == "operator":
        decision = {"action": action}
        if action == "edit":
            decision["text"] = (edited_text or "").strip()
        run_workflow(graph, config, Command(resume=decision))

    return refresh_operator(request)


def approve(thread_id, edited_text, request: gr.Request):
    return operator_decide("approve", thread_id, edited_text, request)


def send_edited(thread_id, edited_text, request: gr.Request):
    return operator_decide("edit", thread_id, edited_text, request)


def reject(thread_id, edited_text, request: gr.Request):
    return operator_decide("reject", thread_id, edited_text, request)


def create_app():
    with gr.Blocks(title="پشتیبانی آنلاین") as app:
        gr.Navbar(visible=False)

        thread_id = gr.State()
        waiting = gr.State(False)

        gr.Markdown("""
        # 🎧 پشتیبانی آنلاین
        ### سؤال یا شکایت خود را بنویسید، ما پاسخگو هستیم
        """, elem_classes=["rtl", "main-title"])

        status_bar = gr.Markdown("💬 آماده دریافت پیام شما", elem_classes="status-bar")

        chatbot = gr.Chatbot(
            label="گفتگو با پشتیبانی",
            height=480,
            elem_classes=["rtl", "chat-container"],
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="پیام شما",
                placeholder="پیام خود را بنویسید... (مثال: سفارشم به شماره ۱ را لغو کنید)",
                scale=4,
                lines=1,
                elem_classes="rtl"
            )
            send_btn = gr.Button("📤 ارسال", scale=1, variant="primary")

        gr.Examples(
            examples=[
                ["می‌خواهم ۲ عدد هدفون انکر سفارش بدهم"],
                ["سفارشم به شماره ۱ را لغو کنید"],
                ["سفارش ۲ من کجاست؟"],
                ["شرایط مرجوعی کالا چیست؟"],
                ["قیمت لپ‌تاپ ایسوس چقدر است و موجود دارید؟"],
                ["چه دسته‌بندی‌هایی از محصولات دارید؟"],
                ["آیا شعبه حضوری در شیراز دارید؟"],
            ],
            inputs=msg_input,
            label="💡 نمونه پیام‌ها"
        )

        with gr.Row():
            new_chat_btn = gr.Button("🔄 گفتگوی جدید", variant="secondary")
            gr.Button("🚪 خروج از حساب", variant="secondary", link="/logout")

        timer = gr.Timer(3)

        chat_outputs = [chatbot, msg_input, status_bar, waiting]

        app.load(fn=start_chat, outputs=[chatbot, thread_id, status_bar, waiting])
        send_btn.click(fn=send_message, inputs=[msg_input, chatbot, thread_id, waiting], outputs=chat_outputs)
        msg_input.submit(fn=send_message, inputs=[msg_input, chatbot, thread_id, waiting], outputs=chat_outputs)
        new_chat_btn.click(fn=start_chat, outputs=[chatbot, thread_id, status_bar, waiting])
        timer.tick(fn=check_operator_reply, inputs=[chatbot, thread_id, waiting], outputs=[chatbot, status_bar, waiting])

    with app.route("پنل اپراتور", "/operator", show_in_navbar=False) as operator_page:
        gr.Markdown("""
        # 🧑‍💼 پنل اپراتور پشتیبانی
        ### تأیید لغو و مرجوعی سفارش، درخواست تماس و پاسخ‌هایی که امتیاز کافی نگرفتند
        """, elem_classes=["rtl", "main-title"])

        with gr.Row():
            with gr.Column(scale=1):
                pending_status = gr.Markdown("", elem_classes="status-bar")
                pending_dropdown = gr.Dropdown(label="🧵 درخواست‌های در انتظار", choices=[], elem_classes="rtl")
                refresh_btn = gr.Button("🔄 به‌روزرسانی", variant="secondary")

            with gr.Column(scale=2):
                review_output = gr.Markdown("", elem_classes="rtl")
                draft_input = gr.Textbox(label="✏️ متن پاسخ (قابل ویرایش)", lines=6, visible=False, elem_classes="rtl")

                with gr.Row():
                    approve_btn = gr.Button("✅ تأیید", variant="primary")
                    edit_btn = gr.Button("✏️ ارسال متن ویرایش‌شده", variant="secondary", visible=False)
                    reject_btn = gr.Button("❌ رد", variant="stop")

        with gr.Accordion("📦 سفارش‌ها", open=False):
            orders_table = gr.Dataframe(
                headers=["شماره", "حساب کاربری", "نام گیرنده", "اقلام", "مبلغ کل (تومان)", "وضعیت", "آخرین تغییر"],
                interactive=False
            )

        with gr.Accordion("📞 درخواست‌های تماس ثبت‌شده", open=False):
            callbacks_table = gr.Dataframe(
                headers=["کد پیگیری", "thread_id", "شماره تماس", "سؤال", "وضعیت", "تاریخ"],
                interactive=False
            )

        operator_outputs = [
            pending_dropdown, review_output, draft_input, edit_btn,
            pending_status, orders_table, callbacks_table
        ]

        operator_page.load(fn=refresh_operator, outputs=operator_outputs)
        refresh_btn.click(fn=refresh_operator, outputs=operator_outputs)
        pending_dropdown.input(fn=show_review, inputs=pending_dropdown, outputs=[review_output, draft_input, edit_btn])
        approve_btn.click(fn=approve, inputs=[pending_dropdown, draft_input], outputs=operator_outputs)
        edit_btn.click(fn=send_edited, inputs=[pending_dropdown, draft_input], outputs=operator_outputs)
        reject_btn.click(fn=reject, inputs=[pending_dropdown, draft_input], outputs=operator_outputs)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        auth=lambda username, password: authenticate(username, password) is not None,
        auth_message="برای استفاده از پشتیبانی آنلاین وارد حساب کاربری خود شوید (کاربران نمونه در README).",
        theme=gr.themes.Soft(),
        css="""
        .rtl {direction: rtl; text-align: right;}
        .rtl textarea, .rtl input {direction: rtl; text-align: right;}
        .chat-container {direction: rtl;}
        .status-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-weight: bold;
            direction: rtl;
        }
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        """
    )
