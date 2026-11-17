import numpy as np

from database import get_faqs, get_categories, get_products
from services import embed_texts

TOP_K = 4

_documents = None
_vectors = None


def catalog_overview():
    categories = "، ".join(c["name"] for c in get_categories())
    products = "، ".join(p["name"] for p in get_products())
    return (
        f"فهرست کامل دسته‌بندی‌های فروشگاه: {categories}\n"
        f"فهرست کامل محصولات فروشگاه: {products}"
    )


def build_documents():
    documents = []

    for faq in get_faqs():
        documents.append(f"سؤال متداول: {faq['question']}\nپاسخ: {faq['answer']}")

    for category in get_categories():
        documents.append(f"دسته‌بندی {category['name']}: {category['description']}")

    for product in get_products():
        stock = f"{product['stock']} عدد موجود" if product["stock"] > 0 else "ناموجود"
        documents.append(
            f"محصول: {product['name']} | دسته‌بندی: {product['category']} | "
            f"قیمت: {product['price']:,} تومان | موجودی: {stock} | {product['description']}"
        )

    return documents


def normalize_rows(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def load_index():
    global _documents, _vectors
    # NOTE: whole index is re-embedded after each order change (reset_index), fine for a small catalog
    if _documents is None:
        documents = build_documents()
        _vectors = normalize_rows(np.array(embed_texts(documents)))
        _documents = documents
    return _documents, _vectors


def reset_index():
    global _documents, _vectors
    _documents = None
    _vectors = None


def retrieve(query, top_k=TOP_K):
    documents, vectors = load_index()
    query_vector = normalize_rows(np.array(embed_texts([query])[0]))
    scores = vectors @ query_vector
    best = np.argsort(scores)[::-1][:top_k]
    # NOTE: full catalog list in every prompt so a "we don't have it" answer is safe, fine for a small catalog
    return [catalog_overview()] + [documents[i] for i in best]
