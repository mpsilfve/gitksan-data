# Gitksan inflection tables

The main resource is the file `gitksan-inflection-tables.txt`. The resource is documented in the LREC 2022 paper:

```
@InProceedings{oliver-EtAl:2022:LREC,
  author    = {Oliver, Bruce  and  Forbes, Clarissa  and  Yang, Changbing  and  Samir, Farhan  and  Coates, Edith  and  Nicolai, Garrett  and  Silfverberg, Miikka},
  title     = {An Inflectional Database for Gitksan},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {6597--6606},
  url       = {https://aclanthology.org/2022.lrec-1.710}
}
```

## Updates

* June 20: Filtered out 117 tables which do not represent noun or verb lexemes 

## TODO

* Some forms have lost their initial "'" (e.g. "'nii" -> "nii"). 
* Some English glosses have orphan square brackets (e.g. "kill["). 
* There are some complex stems with voicing errors (e.g. "nitin" which should be "'nidin").

## License

For now, we release the data lincensed under [Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported (CC BY-NC-ND 3.0)](https://creativecommons.org/licenses/by-nc-nd/3.0/). Please get in touch with the authors if you need a more permissive license.
