#create document for Monday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "MondayAnalysis.md", params = list(dow = "Monday"))