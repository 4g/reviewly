import json

from google_play_scraper import app, reviews
from pathlib import Path
from settings import PathConfig
from tqdm import tqdm

class Fields:
    COMPANY = "company"

class PlayStoreReview(Fields):
    COMMENT = "content"
    RATING = "score"
    REPLY = "replyContent"
    TIMESTAMP = "at"
    HELPFUL_COUNT = "thumbsUpCount"
    ID = "reviewId"

    @staticmethod
    def get(review, field):
        return review[field]


class AppFilter:
    def __init__(self, name, country='in', lang='en'):
        self.name = name
        self.country = country
        self.lang = lang

    def __repr__(self):
        return f"app={self.name} country={self.country} lang={self.lang}"


class ReviewStore:
    def __init__(self):
        self.write_file_pointers = {}

    def get_info(self, appfilter: AppFilter):
        path = get_info_path(appfilter)
        info = json.load(open(path))
        return info

    def get_reviews(self, appfilter: AppFilter):
        path = get_reviews_path(appfilter)
        for line in open(path):
            review = json.loads(line)
            yield review

    def get_appnames(self):
        basepath = Path(PathConfig.reviews_path)
        appnames = []
        for path in basepath.iterdir():
            if path.is_dir():
                name = path.name
                appnames.append(name)

        return appnames

    def add_review(self, appfilter, review):
        path = get_reviews_path(appfilter)
        path.parent.mkdir(exist_ok=True, parents=True)
        fp = self.write_file_pointers.get(path, open(path, 'w'))
        review = json.dumps(review, sort_keys=True, default=str) + "\n"
        fp.write(review + "\n")

    def add_info(self, appfilter, info):
        info_path = get_info_path(appfilter)
        with open(info_path, 'w') as fp:
            json.dump(info, fp, indent=4, default=str)

    def close(self):
        for fname in self.write_file_pointers:
            fp = self.write_file_pointers[fname]
            fp.close()

def get_path(appfilter : AppFilter, fname: str):
    return f"{PathConfig.reviews_path}/{appfilter.name}/{appfilter.country}.{appfilter.lang}.{fname}"


def get_reviews_path(appfilter: AppFilter):
    return Path(get_path(appfilter, "reviews.txt"))


def get_info_path(appfilter: AppFilter):
    return Path(get_path(appfilter, "info.txt"))


def get_info_from_playstore(appfilter: AppFilter):
    result = app(
        appfilter.name,
        lang=appfilter.lang,
        country=appfilter.country
    )

    return result


def download_info_to_file(appfilter: AppFilter):
    print(f"Downloading info {appfilter}")
    info = get_info_from_playstore(appfilter)
    review_store = ReviewStore()
    review_store.add_info(appfilter, info)
    print(f"Downloaded info {appfilter}")
    return info


def stream_reviews_from_playstore(appfilter: AppFilter):
    continuation_token = None
    max_count_fetch = 199

    while True:
        print("Token", continuation_token)
        _result, continuation_token = reviews(
            app_id=appfilter.name,
            count=max_count_fetch,
            continuation_token=continuation_token,
            country=appfilter.country,
            lang=appfilter.lang
        )

        for review in _result:
            yield review

        if continuation_token.token is None:
            break


def download_reviews_to_file(appfilter: AppFilter, force_download: bool = False):
    existing_reviews = list(ReviewStore().get_reviews(appfilter))
    if len(existing_reviews) > 0 and (force_download is False):
        print(f"Path {appfilter} exists, skipping")
        return

    desc = f"Downloading reviews {appfilter}"
    review_store = ReviewStore()
    for review in tqdm(stream_reviews_from_playstore(appfilter), desc=desc):
        review_store.add_review(appfilter, review)

    print(f"Downloaded {appfilter}")

def download_data(appnames):
    country = 'in'
    lang = 'en'
    for appname in appnames:
        appfilter = AppFilter(name=appname, country=country, lang=lang)

        info_path = get_info_path(appfilter)
        info_path.parent.mkdir(exist_ok=True, parents=True)
        download_info_to_file(appfilter)

        review_path = get_reviews_path(appfilter)
        download_reviews_to_file(appfilter)

if __name__ == '__main__':
    # appnames = [line.strip() for line in open("companies.txt")]
    # download_data(appnames)
    x = ReviewStore().get_appnames()
    print(len(x), "Apps found", x)
    for name in x:
        appfilter = AppFilter(name)
        reviews = ReviewStore().get_reviews(appfilter)
        reviews = list(reviews)

        info = ReviewStore().get_info(appfilter)
        print(name, len(reviews), info["reviews"])