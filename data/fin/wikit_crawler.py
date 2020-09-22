import sys, requests

words_fn = sys.argv[1]

words = []
with open(words_fn, encoding="utf-8") as words_f:
    for word_line in words_f:
        word = word_line.strip().split(' ', 1)[1]
        words.append(word)

for word in words:
    word_wikit_resp = requests.get(f"https://sv.wiktionary.org/wiki/{word}")
    print(word, word_wikit_resp.text)
    input()

