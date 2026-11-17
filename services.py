import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, NOT_GIVEN

load_dotenv(Path(__file__).parent / ".env")

AVALAI_BASE_URL = "https://api.avalai.ir/v1"
DEFAULT_MODEL = os.environ.get("AVALAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("AVALAI_EMBEDDING_MODEL", "text-embedding-3-small")

INTENTS = [
    "place_order", "cancel_order", "return_order", "order_status",
    "confirm_yes", "confirm_no", "question", "complaint",
]

CAPABILITIES = """What this chatbot can do by itself inside this chat:
- show the status and items of an order,
- place a new order (after the customer confirms the summary),
- cancel an order that has not been shipped yet (after operator approval),
- register a return request for a delivered order within 7 days of delivery (after operator approval),
- register a callback request when a question has no answer in the knowledge base."""

CLASSIFIER_PROMPT = """You are the intent router of a Persian customer support chatbot for an online shop.
Classify the customer's latest message into exactly one intent:
- "place_order": the customer wants to buy / order products (including references like "order both of them"),
  or is giving the missing details (products, quantities, name) of an order that is in progress.
- "cancel_order": the customer wants to cancel an existing order that has not been delivered (لغو).
- "return_order": the customer wants to return / send back a product or order they received (مرجوع، پس دادن، عودت),
  even when they also express dissatisfaction.
- "order_status": the customer asks where an order is or what its status is.
- "confirm_yes": the customer says yes to the question "Do you want us to call you within 48 hours?".
- "confirm_no": the customer says no to that question.
- "question": the customer asks about products, prices, stock, categories, shipping, payment, returns, warranty or any other information.
- "complaint": the customer complains about a problem or a bad experience.

Only use "confirm_yes" or "confirm_no" when awaiting_callback is true.
If an order is in progress (cart_in_progress is true) and the message gives order details, the intent is "place_order".
Read the recent conversation to understand short answers and references to earlier messages.

Extract these fields (null when not known; convert Persian digits to English):
- "order_id": the id of the existing order the customer means (integer), for cancel_order, return_order or
  order_status. Resolve references like "it", "همون سفارش", "مرجوعش کن" from the recent conversation;
  active_order_id is the order that was being discussed.
- "phone": the customer's phone number as a string of digits.

Return only JSON: {"intent": "...", "order_id": ..., "phone": ...}"""

ORDER_EXTRACTOR_PROMPT = """You maintain the shopping cart of a Persian online shop chatbot.
Read the WHOLE conversation and return the COMPLETE list of products the customer wants to order now,
after applying their latest message to the current cart.

Rules:
- Resolve references such as "both of them" (جفتش / هر دو), "those two", "the ones you said" from earlier
  messages, including products the assistant listed or recommended.
- "Two products" / "both" means two DIFFERENT products with quantity 1 each, unless a quantity is given per product.
- A number written next to a product is that product's quantity only.
- If the latest message changes the current cart (add, remove, change a quantity, replace a product), apply
  that change and keep the rest of the cart.
- Use only product ids from the catalog. Leave out anything that is not in the catalog.

Product catalog (id: name — price in toman):
{catalog}

Current cart: {cart}

Return only JSON: {{"items": [{{"product_id": <int>, "quantity": <int>}}]}}"""

CONFIRMATION_PROMPT = """A Persian online shop chatbot showed the customer this order summary and asked
whether it should place the order:

{summary}

Classify the customer's reply:
- "confirm": the customer agrees to place exactly this order (e.g. "بله", "آره ثبت کن", "درسته").
- "cancel": the customer does not want to place any order now (e.g. "نه", "منصرف شدم").
- "modify": the customer wants to change anything (products, quantities, name), even if they also say yes
  (e.g. "بله ولی هدفون دو تا باشد"), or the reply is about something else.

Return only JSON: {{"decision": "confirm" | "cancel" | "modify"}}"""

BASE_RULES = """The reply MUST be written in fluent, natural Persian (Farsi). Do not use English words.
Never make guaranteed or unauthorized promises: no "guarantee", "definitely", "100%", "immediately",
and no compensation or outcome that has not been confirmed yet.
Write one plain-text paragraph, no markdown, no placeholders like [name].
Return only the reply text."""

COMPLAINT_PROMPT = f"""You are a senior customer support agent for an Iranian online shop.
You chat with customers on the support page and reply to their complaints.
Use the earlier messages of the conversation and the knowledge base context when they are relevant.

{CAPABILITIES}

If the customer's problem can be solved by one of these, tell them exactly what to write in this chat
(e.g. "بنویسید سفارش شماره ۳ را مرجوع کنید"). Never invent other procedures, and never ask for information
the customer already gave earlier in the conversation (such as the order number).
Never contradict the knowledge base.

Every reply must have:
- A respectful and empathetic tone (address the customer politely, e.g. "مشتری گرامی", and acknowledge their frustration).
- A clear apology.
- One specific next action (one of the capabilities above, or referring the case to the support team).
- A concrete follow-up time or channel that does not contradict the knowledge base.

Length: 60 to 120 words.
{BASE_RULES}"""

QUESTION_PROMPT = f"""You are a customer support agent for an Iranian online shop.
Answer the customer's question using ONLY the knowledge base context below and the earlier messages of the conversation.
Always start the answer with "مشتری گرامی،" and answer clearly and briefly.

The knowledge base is the COMPLETE and authoritative description of the shop: its full product list, categories,
branches, shipping areas, payment methods, services and policies. Classify your reply with "type":
- "answer": the knowledge base contains the requested information. Give it.
- "no": the customer asks whether the shop has, sells or offers something (a product, brand, branch, city,
  payment method, shipping option, service) and it is NOT in the knowledge base. Because the knowledge base
  is complete, the shop does not have it: say "no" clearly and mention what the shop does offer instead.
  Example: the knowledge base lists a branch only in Isfahan and the customer asks about Shiraz -> type "no".
  Example: the knowledge base says nothing about repair services -> the shop has no repair service -> type "no".
- "unknown": ONLY when the question asks for a specific fact that is neither in the knowledge base nor
  answerable with "no" (e.g. a future release date, the customer's personal account details, a topic
  unrelated to the shop). Do not guess; leave the answer empty.

Never write vague replies like "no information is available"; use type "unknown" instead.

Return only JSON: {{"type": "answer" | "no" | "unknown", "answer": "<the Persian reply starting with مشتری گرامی، — empty when type is unknown>"}}

Length of the answer: 20 to 100 words.
{BASE_RULES.replace("Return only the reply text.", "")}"""


_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["AVALAI_API_KEY"],
            base_url=AVALAI_BASE_URL,
        )
    return _client


def system_prompt(intent, context):
    prompt = QUESTION_PROMPT if intent == "question" else COMPLAINT_PROMPT
    knowledge = "\n\n".join(context) or "(empty)"
    return f"{prompt}\n\nKnowledge base context (in Persian):\n{knowledge}"


def call_llm(system, messages, model, json_mode=False):
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        response_format={"type": "json_object"} if json_mode else NOT_GIVEN,
        temperature=0.3,
        timeout=60,
    )
    return response.choices[0].message.content.strip()


def ask(messages, context, intent, model):
    if intent != "question":
        return {"found": True, "answer": call_llm(system_prompt(intent, context), messages, model)}

    result = json.loads(call_llm(system_prompt(intent, context), messages, model, json_mode=True))
    answer = (result.get("answer") or "").strip()
    return {"found": result.get("type") != "unknown" and bool(answer), "answer": answer}


def embed_texts(texts):
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def to_int(value):
    return int(value) if str(value).isdigit() else None


def call_json(system, messages, model):
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        response_format={"type": "json_object"},
        temperature=0,
        timeout=60,
    )
    return json.loads(response.choices[0].message.content)


def format_conversation(messages):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def classify_message(message, awaiting_callback, cart_in_progress, active_order_id, recent_messages, model=DEFAULT_MODEL):
    context = f"""awaiting_callback: {str(awaiting_callback).lower()}
cart_in_progress: {str(cart_in_progress).lower()}
active_order_id: {active_order_id or 'null'}

Recent conversation:
{format_conversation(recent_messages) or '(empty)'}

Latest customer message: {message}"""

    result = call_json(CLASSIFIER_PROMPT, [{"role": "user", "content": context}], model)
    intent = result.get("intent")
    return {
        "intent": intent if intent in INTENTS else "question",
        "order_id": to_int(result.get("order_id")),
        "phone": result.get("phone") or None,
    }


def extract_order(messages, catalog, cart, model=DEFAULT_MODEL):
    system = ORDER_EXTRACTOR_PROMPT.format(catalog=catalog, cart=json.dumps(cart, ensure_ascii=False))
    result = call_json(system, messages, model)
    items = [
        {"product_id": to_int(item.get("product_id")), "quantity": to_int(item.get("quantity")) or 1}
        for item in result.get("items") or []
    ]
    return {"items": [item for item in items if item["product_id"]]}


def interpret_confirmation(summary, answer, model=DEFAULT_MODEL):
    result = call_json(CONFIRMATION_PROMPT.format(summary=summary), [{"role": "user", "content": answer}], model)
    decision = result.get("decision")
    return decision if decision in ("confirm", "cancel", "modify") else "modify"


def generate_draft(messages, context, intent, model=DEFAULT_MODEL):
    return ask(messages, context, intent, model)


def revise_draft(complaint, draft, feedback, context, intent, model=DEFAULT_MODEL):
    problems = "\n".join(f"- {item}" for item in feedback)
    prompt = f"""Customer message (in Persian):
{complaint}

Previous reply:
{draft}

A quality reviewer found these problems in the previous reply:
{problems}

Rewrite the reply in Persian so that every problem is fixed, while keeping what was already good."""
    return ask([{"role": "user", "content": prompt}], context, intent, model)["answer"]
