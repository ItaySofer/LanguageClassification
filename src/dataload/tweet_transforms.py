from urllib.parse import urlparse


class SplitToWords(object):

    def __init__(self):
        pass

    def __call__(self, tweet):
        return tweet.split(" ")


class RemoveURLs(object):

    def __init__(self):
        pass

    def __call__(self, word_list):
        return [word for word in word_list if RemoveURLs.is_url(word) is False]

    @staticmethod
    def is_url(s):
        parseResult = urlparse(s)
        return parseResult.scheme != '' or parseResult.netloc != ''


class RemoveMentions (object):

    def __init__(self):
        pass

    def __call__(self, word_list):
        return [word for word in word_list if RemoveMentions.is_mention(word) is False]

    @staticmethod
    def is_mention(s):
        return s.startswith('@')

