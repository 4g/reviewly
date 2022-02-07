import textlib
from tqdm import tqdm
from pathlib import Path
from itertools import groupby
from embedders import CachedEmbedder, EmbedderFactory

from review_store import ReviewStore, AppFilter, PlayStoreReview

def batch_generator(elements, batch_size):
    for key, elem_batch in tqdm(groupby(enumerate(elements), lambda x: x[0] // batch_size), total=len(elements)//batch_size):
        batch = [l[1] for l in elem_batch]
        yield batch

def embed_sentences(appname, embedder_name):
    review_store = ReviewStore()
    embedder = EmbedderFactory.get_embedder(embedder_name)
    #
    # if embedder is None:
    #     print("Embedder one of : ", EmbedderFactory.ALL)
    #     return
    #
    # if appname not in review_store.get_appnames():
    #     print("App name in :", review_store.get_appnames())
    #     return
    output_name = appname + "-" + embedder_name
    print("Output name", output_name)
    embedding_store = CachedEmbedder(embedder=embedder(), name=output_name)

    comments = set()

    appfilter = AppFilter(name=appname)

    print("Using app", appfilter)
    reviews = review_store.get_reviews(appfilter=appfilter)
    text_processor = textlib.TextProcessor()

    for review in reviews:
        comment = PlayStoreReview.get(review, PlayStoreReview.COMMENT)
        if not comment:
            continue
        sentences = text_processor.split_para_to_sentences(comment)
        if sentences:
            for sentence in sentences:
                comments.add(sentence)

    comments = sorted(list(set(comments)))
    print(f"Unique comments {len(comments)}")

    for comment in batch_generator(comments, 64):
        embedding_store.embed(comment)

    print("cache hits", embedding_store.get_cache_hit_ratio())
    print("pending commits", embedding_store.get_pending_commit_size())
    embedding_store.commit()


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--appname", help=f"pass one of the apps from {ReviewStore().get_appnames()}", required=True)
    parser.add_argument("--embedder", help=f"pass one of embedders from {EmbedderFactory.list_all()}" , required=True)

    args = parser.parse_args()
    embed_sentences(args.appname, args.embedder)