#create document for Monday
rmarkdown::render(input = "ST558Project2.Rmd", output_format = "github_document", output_file = "SaturdayAnalysis.md", params = list(dow = "Saturday"))

weekday <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


#read in data
dayData <- read_csv("day.csv")
dayData

#get unique days / might need to convert dayIDs into literal days to match the output link
dayIDs <- unique(dayData$status)

output_file = paste0(dayIDs, "Analysis.md")
params <- lapply(weekday, FUN = function(x){list(dow = x)})

#put into a data frame 
reports <- tibble(output_file, params)
reports

apply(documents, MARGIN = 1, FUN = function(x){
  rmarkdown::render(input = "ST558Project2.Rmd", output_file = x[[1]], params = x[[2]])
}
)

