library(tidyverse)
library(patchwork)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

estimator_lookup <- tribble(
  ~estimator,        ~label,
  "outcome_adapted", "Outcome-adapted\nNeural Net",
  "0.01",            "RieszNet,\nlambda_Riesz = 0.01",
  "0.1",             "RieszNet,\nlambda_Riesz = 0.1",
  "1",               "RieszNet,\nlambda_Riesz = 1",
  "10",              "RieszNet,\nlambda_Riesz = 10",
  "100",             "RieszNet,\nlambda_Riesz = 100",
  "riesz_adapted",   "Riesz-adapted\nNeural Net"
)

output_path <- "plots/varying_riesz_net_weights.pdf"

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

read_experiment <- function(path, estimator_col) {
  read_csv(path, show_col_types = FALSE) %>%
    mutate(estimator = as.character(.data[[estimator_col]])) %>%
    filter(estimator %in% estimator_lookup$estimator) %>%
    select(point_estimate, truth, estimator, seed)
}

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

dope_nets <- read_experiment(
  path = "results/varying_weights_plots_experiments/ihdp_outcome_vs_riesz_adapted.csv",
  estimator_col = "training_procedure"
)

riesz_net <- read_experiment(
  path = "results/varying_weights_plots_experiments/ihdp_varying_riesz_weight.csv",
  estimator_col = "riesz_weight"
)

tmle_net <- read_csv(
  "results/varying_weights_plots_experiments/ihdp_varying_tmle_weight.csv",
  show_col_types = FALSE
)

# -------------------------------------------------------------------------
# Summarise results
# -------------------------------------------------------------------------

riesz_plot_data <- bind_rows(
  dope_nets,
  riesz_net
) %>%
  mutate(error = point_estimate - truth) %>%
  group_by(estimator) %>%
  summarise(
    mse = mean(error^2),
    mse_se = sd(error^2) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    mse_lower = mse - 1.96 * mse_se,
    mse_upper = mse + 1.96 * mse_se,
    estimator = factor(estimator, levels = estimator_lookup$estimator)
  ) %>%
  arrange(estimator)

tmle_plot_data <- tmle_net %>%
  mutate(error = point_estimate - truth) %>%
  group_by(tmle_weight) %>%
  summarise(
    mse = mean(error^2),
    count = n(),
    mse_se = sd(error^2) / sqrt(count),
    .groups = "drop"
  ) %>%
  mutate(
    mse_lower = mse - 1.96 * mse_se,
    mse_upper = mse + 1.96 * mse_se
  )

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

riesz_weights <- ggplot(riesz_plot_data, aes(x = estimator, y = mse, group = 1)) +
  geom_line(linetype = "dashed") +
  geom_point() +
  geom_errorbar(
    aes(ymin = mse_lower, ymax = mse_upper),
    width = 0.15
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0.05, 0.15))
  ) +
  scale_x_discrete(
    labels = set_names(estimator_lookup$label, estimator_lookup$estimator),
    guide = guide_axis(angle = 45)
  ) +
  labs(
    x = "Estimator",
    y = "MSE",
    title = "RieszNet MSE with Varying lambda_Riesz"
  ) +
  theme_classic(base_size = 16) +
  theme(
    plot.margin = margin(t = 20, r = 60, b = 10, l = 10),
    plot.title = element_text(hjust = 0.5),
    plot.title.position = "panel"
  )

tmle_weights <- ggplot(tmle_plot_data, aes(x = tmle_weight, y = mse)) +
  geom_line(linetype='dashed') +
  geom_point() +
  geom_errorbar(
    aes(ymin = mse_lower, ymax = mse_upper),
    width = 0.15
  ) +
  scale_y_continuous(
    limits = c(0, 0.04),
    expand = expansion(mult = c(0.05, 0.05))
  ) +
  scale_x_continuous(
    breaks = sort(unique(tmle_plot_data$tmle_weight)),
    labels = scales::number_format(accuracy = 0.1)
  ) +
  labs(
    x = "lambda_TMLE",
    y = "MSE",
    title = "RieszNet MSE with Varying lambda_TMLE"
  ) +
  theme_classic(base_size = 16) +
  theme(
    plot.margin = margin(t = 20, r = 10, b = 10, l = 10),
    plot.title = element_text(hjust = 0.5),
    plot.title.position = "panel"
  )

varying_weights <- 
  free(riesz_weights, side = "b") + 
  tmle_weights

ggsave(
  filename = output_path,
  plot = varying_weights,
  width = 12,
  height = 6,
  create.dir = TRUE
)
