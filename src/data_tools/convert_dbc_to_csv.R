## Script to convert dbc files to csv ##

# install.packages("read.dbc")
library(read.dbc)

data_dir <- "~/Documents/GitHub/Imperial/Dengue-Nowcasting-Thesis/data/raw/counts/"

years <-  seq(12, 23)

for(year in years){
  file_name <- paste0("DENGBR", year)
  read_file_path <- paste0(data_dir, file_name, ".dbc")
  write_file_path <- paste0(data_dir, file_name, ".csv")
  
  
  data <- read.dbc(read_file_path)
  write.csv(data, write_file_path, row.names = FALSE)
  
}





