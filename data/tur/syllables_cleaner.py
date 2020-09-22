import re
import sys
from collections import defaultdict

syllabified_words_fn = sys.argv[1]
orig_words_fn = sys.argv[2]

orig_words = {line.strip() for line in open(orig_words_fn, encoding="utf-8")}

word2syllables = defaultdict()
with open(syllabified_words_fn, encoding="utf-8") as syl_f:
    for line in syl_f:
        if not line.strip() or line.strip().split()[0] not in orig_words:
            continue
        word, syllables = line.strip().split(maxsplit=1)
        word2syllables[word] = syllables

for w in sorted(word2syllables.keys()):
    print(w, word2syllables[w])
