#' @keywords internal
"_PACKAGE"

dbhdistfit <- NULL

.onLoad <- function(libname, pkgname) {
  dbhdistfit <<- reticulate::import("dbhdistfit", delay_load = TRUE)
}
