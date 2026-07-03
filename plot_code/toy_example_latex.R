library(tidyverse)
library(ggnewscale)
source("plot_code/latex_plot_utils.R")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

n <- 500

input_path <- "results/toy_example_plot_experiments/toy_example.csv"
output_path <- "plots/toy_example.pdf"

representation_colors <- c(
  "Z" = "#F8766D",
  "X" = "#00BFC4"
)

representation_labels <- c(
  "Z" = "$Z = U$",
  "X" = "$X = (W, U)$"
)

estimator_colors <- c(
  "dope_net" = "#F8766D",
  'dope_net_no_dim_reduction' = "#7C9B99",
  "separate_nets" = "#00BFC4"
)

estimator_labels <- c(
  "dope_net" = "Outcome-adapted Neural Network \nw. Adaptive Dimensionality Reduction",
  'dope_net_no_dim_reduction' = 'Outcome-adapted Neural Network \nw. No Dimensionality Reduction',
  "separate_nets" = "Separate Neural Networks"
)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

asymptotic_variance <- function(representation, beta) {
  case_when(
    representation == "Z" | beta == 0 ~ 4,
    representation == "X" ~ (exp(beta) - exp(-beta)) / beta + 2
  )
}

# -------------------------------------------------------------------------
# Load and summarise data
# -------------------------------------------------------------------------

asymptotic_variance_df <- expand_grid(
  beta = seq(0, 4, length.out = 1000),
  representation = c("Z", "X")
) %>%
  mutate(asymptotic_variance = asymptotic_variance(representation, beta))

simulation_df <- read_csv(input_path, show_col_types = FALSE)

mse_df <- simulation_df %>%
  mutate(error = point_estimate - truth) %>%
  group_by(beta, model) %>%
  summarise(
    mse = mean(error^2),
    se = sd(error^2) / sqrt(n()),
    count = n(),
    .groups = "drop"
  ) %>%
  mutate(
    mse = n * mse,
    se = n * se,
    lower = mse - 1.96 * se,
    upper = mse + 1.96 * se
  )

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

toy_plot <- ggplot() +
  geom_line(
    data = asymptotic_variance_df,
    aes(x = beta, y = asymptotic_variance, color = representation)
  ) +
  scale_color_manual(
    name = "Representation",
    values = representation_colors,
    labels = representation_labels
  ) +
  ggnewscale::new_scale_color() +
  geom_errorbar(
    data = mse_df,
    aes(x = beta, ymin = lower, ymax = upper, color = model),
    width = 0.08
  ) +
  geom_point(
    data = mse_df,
    aes(x = beta, y = mse, color = model)
  ) +
  scale_color_manual(
    name = "AutoDML estimator",
    values = estimator_colors,
    labels = estimator_labels
  ) +
  labs(
    x = "$\\beta$",
    y = "$n\\cdot\\mathrm{MSE}$",
    title = "Estimator and Asymptotic Mean-square Error"
  ) +
  theme_classic(base_size = latex_base_size, base_family = latex_font_family) +
  theme(
    plot.margin = latex_plot_margin
  ) +
  latex_title_theme +
  ylim(0,22)

toy_plot

save_latex_plot(
  filename = output_path,
  plot = toy_plot,
  width = latex_single_width,
  height = latex_single_height
)
