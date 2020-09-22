WORDS_F=$1

# grep -ir -o "syllabification:.*$"  <(lynx https://en.wiktionary.org/wiki/laatusana --dump)



for word in $(cat $WORDS_F | sed "s|[[:space:]]*[[:digit:]]*[[:space:]]*\(.*\)$|\1|g"); do
	syllables=$(curl -v --silent -XGET \
                https://ru.wiktionary.org/wiki/$word 2>&1 \
                | grep hyph \
                | perl -pe "s|.*(<b>.*?</b>).*|\1|" \
		| perl -pe "s|[[:space:]]*<span class=\"hyph\" style=\"color:lightgreen;\">-</span>[[:space:]]*|‧|g" \
                | perl -pe "s|[[:space:]]*<span class=\"hyph-dot\" style=\"color:red;\">·</span>[[:space:]]*|‧|g" \
		| perl -pe "s|.*<b>(.*)</b>.*|\1|" \
		| head -1)
	if [ ! -z "$syllables" ]; then echo $word "$syllables"; fi;
done;

