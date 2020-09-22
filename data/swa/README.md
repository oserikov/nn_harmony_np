data taken from https://korp.csc.fi/download/HCS/a-v2/.zip/hcs-a-v2-dl.zip

clean texts consist of top N open class words

early version of open class words extraction is smth like `grep -v ^\< hcs2_old_news.vrt | cut -d $'\t' -f1,3 | grep [[:space:]][N,V,ADJ]$ | cut -d $'\t' -f1 | sed -e 's/.*/\L\0/'`
