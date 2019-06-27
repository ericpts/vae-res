(TeX-add-style-hook
 "refs"
 (lambda ()
   (LaTeX-add-bibitems
    "bib:vae_paper"
    "bib:monet"
    "bib:iodine"
    "bib:betavae"
    "bib:spatial-broadcast-decoder"
    "bib:fashion-mnist"
    "bib:clevr"))
 :bibtex)

