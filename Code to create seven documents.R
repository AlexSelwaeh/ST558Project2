#create document for Monday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "MondayAnalysis.md", params = list(dow = "Monday"))

#create document for Tuesday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "TuesdayAnalysis.md", params = list(dow = "Tuesday"))

#create document for Wednesday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "WednesdayAnalysis.md", params = list(dow = "Wednesday"))

#create document for Thursday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "ThursdayAnalysis.md", params = list(dow = "Thursday"))

#create document for Friday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "FridayAnalysis.md", params = list(dow = "Friday"))

#create document for Saturday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "SaturdayAnalysis.md", params = list(dow = "Saturday"))

#create document for Sunday
rmarkdown::render(input = "ST558Project2.rmd", output_format = "github_document", output_file = "SundayAnalysis.md", params = list(dow = "Sunday"))

library(rmarkdown)
weekday <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
render(input = "ST558Project2.rmd", output_format = "github_document", output_file = paste0(weekday, "Analysis.md"), params = list(dow = weekday))