There four datasets (each subdirectory) in this directory can be divided into two categories. 
One category (`w_root` suffix) includes both inflection data (`LEMMA\tTARGET_FORM\t\SRC_TAG;TGT_TAG`) and re-inflection data (`SRC_FORM\tTARGET_FORM\t\SRC_TAG;TGT_TAG`). 
The other category (no `w_root` suffix) includes **only** re-inflection data. 

The two `random_split` directories comprise a standard 80/10/10 train/dev/test split on the word forms. 
The latter two `seen_unseen_split` directories comprise a 70/10/10/10 train/dev/seen_test/unseen_test split. The `seen_test_set` simply
means that the forms in that dataset are from paradigms that were seen during training, so it should be somewhat easier to do well on. 

**NOTE**: for the directories with the `w_root` suffix (i.e., containing both inflection and re-inflection data), I selected the first root in the paradigms
as the representative lemma. This is worth mentioning since there are some tables with multiple root entries (e.g., see the paradigm on line `35706` in `whitespace-inflection-tables-gitksan-productive.txt`).