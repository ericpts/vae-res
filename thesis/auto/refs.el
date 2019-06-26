(TeX-add-style-hook
 "refs"
 (lambda ()
   (LaTeX-add-bibitems
    "bib:vae_paper"
    "bib:monet"
    "bib:iodine"
    "bib:betavae"))
 :bibtex)

