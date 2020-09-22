WORDS_F=$1

# grep -ir -o "syllabification:.*$"  <(lynx https://en.wiktionary.org/wiki/laatusana --dump)



for word in $(cat $WORDS_F | sed "s|[[:space:]]*[[:digit:]]*[[:space:]]*\(.*\)$|\1|g"); do
	syllables=$(grep -i finnish -A 1000 \
	        <(lynx https://en.wiktionary.org/wiki/$word --dump) \
		| grep -i -o -m1 "\(syllabification\|hyphenation\):.*$" \
		| sed "s|^.*\:[[:space:]]\(.*\)$|\1|gi" | head -1)
	if [ ! -z $syllables ]; then echo $word $syllables; else echo "not found $word" 1>&2; fi;
done;

