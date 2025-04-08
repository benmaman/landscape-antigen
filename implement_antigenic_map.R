
library('Racmacs')


path_to_titer_file <- system.file("extdata/h3map2004_hitable.csv", package = "Racmacs")
path_to_titer_file <- "titer_data_for_antigenic_map.csv"






titer_table        <- read.titerTable(path_to_titer_file)
map <- acmap(
  titer_table = titer_table
)





map <- optimizeMap(
  map                     = map,
  number_of_dimensions    = 2,
  number_of_optimizations = 5000,
  minimum_column_basis    = "none"
)

ag_coords <- agCoords(map)
sr_coords <- srCoords(map)

# Add antigen names
ag_coords_df <- as.data.frame(ag_coords)
ag_coords_df$antigen <- agNames(map)

# Add serum names
sr_coords_df <- as.data.frame(sr_coords)
sr_coords_df$serum <- srNames(map)


# Save to files
write.csv(ag_coords_df, "antigen_coordinates.csv", row.names = FALSE)
write.csv(sr_coords_df, "serum_coordinates.csv", row.names = FALSE)
