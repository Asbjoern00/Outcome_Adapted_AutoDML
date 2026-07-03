latex_font_family <- ""
latex_base_size <- 16
latex_annotation_size <- 4

latex_stacked_width <- 12
latex_stacked_height <- 10
latex_single_width <- 12
latex_single_height <- 6

latex_plot_margin <- ggplot2::margin(t = 20, r = 10, b = 10, l = 10)
latex_right_legend_margin <- ggplot2::margin(t = 20, r = 60, b = 10, l = 10)

latex_title_theme <- ggplot2::theme(
  plot.title = ggplot2::element_text(hjust = 0.5),
  plot.title.position = "panel"
)

latex_percent_format <- function(x) {
  paste0(scales::number(100 * x, accuracy = 1), "\\%")
}

latex_compiler <- function() {
  candidates <- c(
    getOption("tikzLatex"),
    Sys.getenv("R_PDFLATEXCMD"),
    Sys.getenv("R_LATEXCMD"),
    Sys.which("pdflatex")
  )

  candidates[nzchar(candidates)][1]
}

compile_latex_plot <- function(tex_file) {
  compiler <- latex_compiler()

  if (is.na(compiler) || !nzchar(compiler)) {
    stop(
      "No LaTeX compiler found. Install MacTeX or TinyTeX and make pdflatex ",
      "available on PATH, or set R_PDFLATEXCMD/R_LATEXCMD.",
      call. = FALSE
    )
  }

  old_wd <- setwd(dirname(tex_file))
  on.exit(setwd(old_wd), add = TRUE)

  output <- system2(
    compiler,
    args = c("-interaction=nonstopmode", "-halt-on-error", basename(tex_file)),
    stdout = TRUE,
    stderr = TRUE
  )

  if (!identical(attr(output, "status"), NULL)) {
    cat(output, sep = "\n")
    stop("LaTeX compilation failed for ", tex_file, call. = FALSE)
  }
}

purge_latex_intermediates <- function(tex_file) {
  tex_stem <- tools::file_path_sans_ext(tex_file)
  file.remove(paste0(tex_stem, c(".aux", ".log", ".tex")))
}

save_latex_plot <- function(filename, plot, width, height) {
  if (!suppressWarnings(requireNamespace("tikzDevice", quietly = TRUE))) {
    stop("Install the R package tikzDevice to write compiled TeX plots.", call. = FALSE)
  }

  compiler <- latex_compiler()
  if (is.na(compiler) || !nzchar(compiler)) {
    stop(
      "No LaTeX compiler found. Install MacTeX or TinyTeX and make pdflatex ",
      "available on PATH, or set R_PDFLATEXCMD/R_LATEXCMD.",
      call. = FALSE
    )
  }

  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)

  tex_file <- paste0(tools::file_path_sans_ext(filename), ".tex")
  tikzDevice::tikz(
    file = tex_file,
    width = width,
    height = height,
    standAlone = TRUE,
    sanitize = FALSE
  )
  print(plot)
  dev.off()

  compile_latex_plot(tex_file)
  purge_latex_intermediates(tex_file)
}
