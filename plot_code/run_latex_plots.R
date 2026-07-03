args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
  repo_root <- normalizePath(file.path(dirname(script_path), ".."))
  setwd(repo_root)
}

source("plot_code/latex_plot_utils.R")

if (is.na(latex_compiler()) || !nzchar(latex_compiler())) {
  stop(
    "No LaTeX compiler found. Install MacTeX or TinyTeX and make pdflatex ",
    "available on PATH, or set R_PDFLATEXCMD/R_LATEXCMD.",
    call. = FALSE
  )
}

plot_scripts <- c(
  "plot_code/mmo_plot_latex.R",
  "plot_code/ase_plot_latex.R",
  "plot_code/bootstrap_plot_latex.R",
  "plot_code/toy_example_latex.R",
  "plot_code/mae_plot_latex.R",
  "plot_code/varying_riesz_net_weights_latex.R"
)

for (plot_script in plot_scripts) {
  message("Running ", plot_script)
  sys.source(plot_script, envir = new.env(parent = globalenv()))
}
