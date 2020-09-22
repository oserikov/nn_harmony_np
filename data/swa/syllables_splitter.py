import sys
import re
import errno

vow_p = "[aeiou]"
cons_p = "[bcdfghjklmnprstvwyz]"

syllable_p = f"{cons_p}?{cons_p}?{vow_p}"
word_p = f"^({syllable_p})+$"

for line in sys.stdin:
    try:
        line = line.strip()
        if re.match(word_p, line):
            print(line, '·'.join(re.findall(syllable_p, line)))
    except Exception as e:
        if e.errno == errno.EPIPE:
            exit(0)
        else:
            raise(e)
