library(tidyverse)
source("plot_code/latex_plot_utils.R")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

estimator_lookup <- tribble(
  ~estimator,        ~label,
  "dope_net_lasso", "Outcome-adapted Neural Net\nw. Adaptive Dimensionality Reduction",
  "MadNet",   "MADNet\nHines and Hines (2025)",
  'c-net', 'Neural Network C-Learner\nCai et al. (2025)',
  "dope_net_lasso_simple", "Outcome-adapted Neural Net\nw. Simple Dimensionality Reduction",
  "outcome_adapted",   "Outcome-adapted Neural Net",
  "RieszNet",          "RieszNet\nChernozhukov et al. (2022a)",
  "separate_nets",     "Separate Neural Nets",
)

output_path <- "plots/mae_plot.pdf"

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

read_experiment <- function(path, estimator_value = NULL, estimator_col = NULL) {
  data <- read_csv(path, show_col_types = FALSE)

  if (!is.null(estimator_value)) {
    data <- data %>%
      mutate(estimator = estimator_value)
  }

  if (!is.null(estimator_col)) {
    data <- data %>%
      mutate(estimator = .data[[estimator_col]])
  }

  data %>%
    select(point_estimate, truth, estimator, seed)
}

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

dope_net_lasso <- read_experiment(
  path = "results/mae_plot_experiments/ihdp_lambda_lasso_cross_validated.csv",
  estimator_value = "dope_net_lasso"
)

dope_net_lasso_simple <- read_experiment(
  path = "results/mae_plot_experiments/ihdp_representation_size_cross_validated.csv",
  estimator_value = "dope_net_lasso_simple"
)

outcome_adapted <- read_experiment(
  path = "results/mae_plot_experiments/ihdp_outcome_adapted.csv",
  estimator_value = "outcome_adapted"
)

separate_nets <- read_experiment(
  path = "results/mae_plot_experiments/ihdp_separate_nets.csv",
  estimator_value = "separate_nets"
)

riesz_net_ours <- read_experiment(
  path = "results/mae_plot_experiments/ihdp_riesz_net.csv",
  estimator_value = "Our RieszNet"
)

mad_net_ours <- read_experiment(
  path = "results/mae_plot_experiments/ihdp_mad_net.csv",
  estimator_value = "Our MadNet"
)

riesz_net <- tibble(
  estimator = c("RieszNet"),
  mae = c(0.11),
  mae_se = c(0.003)
)

mad_net <- tibble(
  estimator = "MadNet",
  mae = 0.094,
  mae_se = 0.002
)

c_learner <- tibble(
  estimator = c('c-net'),
  mae = c(0.098),
  mae_se = c(0.002)
)

# -------------------------------------------------------------------------
# Summarise results
# -------------------------------------------------------------------------

plot_data <- bind_rows(
  dope_net_lasso,
  dope_net_lasso_simple,
  outcome_adapted,
  separate_nets
  ) %>%
mutate(error = point_estimate - truth) %>%
  group_by(estimator) %>%
  summarise(
    mae = mean(abs(error)),
    mae_se = sd(abs(error)) / sqrt(n()),
    .groups = "drop"
  ) %>%
  bind_rows(riesz_net, mad_net, c_learner) %>%
  mutate(
    mae_lower = mae - 1.96 * mae_se,
    mae_upper = mae + 1.96 * mae_se,
    mae_label = sprintf("%.3f", mae),
    estimator = factor(estimator, levels = estimator_lookup$estimator)
  ) %>%
  arrange(estimator)

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

mae_plot <- ggplot(plot_data, aes(x = estimator, y = mae, group = 1)) +
  geom_point() +
  geom_errorbar(
    aes(ymin = mae_lower, ymax = mae_upper),
    width = 0.15
  ) +
  geom_text(
    aes(y = mae_upper, label = mae_label),
    vjust = -0.6,
    size = latex_annotation_size
  ) +
  scale_x_discrete(
    labels = set_names(estimator_lookup$label, estimator_lookup$estimator)
  ) +
  scale_y_continuous(
    limits = c(0, 0.12),
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  labs(
    x = "Estimator",
    y = "Mean Absolute Error"
  ) +
  theme_classic(base_size = latex_base_size, base_family = latex_font_family)+
  theme(
    axis.text.x = element_text(angle = -25, hjust = 0, vjust = 1),
    plot.margin = latex_right_legend_margin
  ) +
  latex_title_theme +
  ggtitle('Mean Absolute Error of Doubly Robust Estimators')

mae_plot

save_latex_plot(
  filename = output_path,
  plot = mae_plot,
  width = latex_single_width,
  height = latex_single_height
)
