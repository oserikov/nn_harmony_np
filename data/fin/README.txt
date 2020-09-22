finnish data
taken from https://1000mostcommonwords.com/1000-most-common-finnish-words/
TODO: replace with data from apertium



 1902  cat UD_Finnish-TDT/*conllu | grep -v "^#" | grep -vP "^\s*$" | cut -d $'\t' -f2,4 | grep "\(NOUN\|VERB\|ADJ\)$" | sed "s|^\(.*\)\t.*|\L\1|g" | sed "s|^\t||g" | sort | uniq -c | sort -nr | head -10000 > fin_words_clean_10000.txt
