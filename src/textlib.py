class TextProcessor:
    def __init__(self, lower=True, normalize=True, split_on_break=True):
        self.lower = lower
        self.normalize = normalize
        self.split_on_break = split_on_break
        self.valid_chars = set("abcdefghijklmnopqrstuvwxyz. ")

    def split_para_to_sentences(self, paragraph):
        paragraph = paragraph.lower()
        norm_para = ""
        for char in paragraph:
            if char in self.valid_chars:
                norm_para += char
                
        sentences = norm_para.split(".")
        process_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            words = sentence.split()
            words = list(filter(lambda x: x, words))
            sentence = " ".join(words)
            if sentence:
                process_sentences.append(sentence)

        return process_sentences


