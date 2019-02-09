from urllib.parse import urlparse
from abc import abstractmethod


class SplitToWords:
    def __call__(self, tweet):
        return tweet.split(" ")


class JoinWordsToSentence:
    def __call__(self, token_list):
        return ' '.join(token_list)


class RemoveTokenBase:
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


class RemoveNames:
    def __call__(self, token_list):
        return [token for i, token in enumerate(token_list) if self.should_remove(token_list, i) is False]

    @staticmethod
    def should_remove(token_list, token_location):
        token = token_list[token_location]
        return RemoveNames.starts_with_upper(token) and \
               RemoveNames.is_first_token_in_sentence(token_list, token_location) is False

    @staticmethod
    def starts_with_upper(token):
        return len(token) > 0 and token[0].isupper()

    @staticmethod
    def is_first_token_in_sentence(token_list, token_location):
        return token_location == 0 or RemoveNames.is_last_token_in_sentence(token_list, token_location - 1)

    @staticmethod
    def is_last_token_in_sentence(token_list, token_location):
        stop_chars = ['.', '?', '!']
        token = token_list[token_location]
        return len(token) > 0 and token[-1] in stop_chars


class CleanTokens:
    def __init__(self):
        pass

    def __call__(self, token_list):
        return [CleanTokens.clean(token) for token in token_list]

    @staticmethod
    def clean(token):
        return ''.join(c for c in token if c.isalpha())


class ToLowerCase:
    def __init__(self):
        pass

    def __call__(self, token_list):
        return [token.lower() for token in token_list]

