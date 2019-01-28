from urllib.parse import urlparse
from abc import abstractmethod
import emoji


class SplitToWords(object):
    def __init__(self):
        pass

    def __call__(self, tweet):
        return tweet.split(" ")


class RemoveTokenBase(object):
    def __init__(self):
        pass
    
    def __call__(self, token_list):
        return [token for token in token_list if self.should_remove(token) is False]

    @abstractmethod
    def should_remove(self, token):
        pass


class RemoveURLs(RemoveTokenBase):
    def should_remove(self, token):
        parseResult = urlparse(token)
        return parseResult.scheme != '' or parseResult.netloc != ''


class RemoveMentions(RemoveTokenBase):
    def should_remove(self, token):
        return token.startswith('@')


class RemoveResponseToken(RemoveTokenBase):
    def should_remove(self, token):
        return token == "RT"


class RemoveHashtags(RemoveTokenBase):
    def should_remove(self, token):
        return token.startswith('#')


class RemoveBlanks(RemoveTokenBase):
    def should_remove(self, token):
        return token == ""


class RemoveEmojis(RemoveTokenBase):
    def should_remove(self, token):
        return all(c in emoji.UNICODE_EMOJI for c in token)

