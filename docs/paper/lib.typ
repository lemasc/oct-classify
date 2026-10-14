// Draft markers, visible in the PDF until resolved.
#let todo(body) = text(fill: red)[[TODO: #body]]
#let pending(body) = text(fill: orange)[[pending: #body]]
