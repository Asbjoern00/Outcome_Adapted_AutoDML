library(tidyverse)
library(patchwork)
source("plot_code/latex_plot_utils.R")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

estimator_lookup <- tribble(
  ~estimator,                       ~label,
  "dope_net_lasso_lambda",          "Outcome-adapted Neural Net\nw. Adaptive Dimensionality Reduction",
  "dope_net_representation_size",   "Outcome-adapted Neural Net\nw. Simple Dimensionality Reduction",
  "dope_net",                       "Outcome-adapted Neural Net",
  "riesz_net",                      "RieszNet",
  "mad_net",                        "MadNet",
  "separate_nets",                  "Separate Neural Nets"
)

ase_output_path <- "plots/ase_plot.pdf"

# Shared color scale used in both plots
estimator_color_scale <- scale_color_discrete(
  name = "Estimator",
  breaks = estimator_lookup$estimator,
  labels = estimator_lookup$label,
  drop = FALSE
)

# -------------------------------------------------------------------------
# Load and summarise results
# -------------------------------------------------------------------------

results_data <- read_csv(
  "results/ase_plot_experiment/experiment.csv",
  show_col_types = FALSE
) %>%
  filter(model %in% estimator_lookup$estimator) %>%
  mutate(
    error = point_estimate - truth,
    model = factor(model, levels = estimator_lookup$estimator)
  )

decomposition_metric_levels <- c(
  "scaled_bias_squared",
  "scaled_variance",
  "scaled_mse"
)

decomposition_metric_labels <- c(
  scaled_bias_squared = "$n\\cdot\\mathrm{Bias}^2$",
  scaled_variance = "$n\\cdot\\mathrm{Variance}$",
  scaled_mse = "$n\\cdot\\mathrm{MSE}$"
)

make_decomposition_plot_data <- function(data) {
  bind_rows(
    data %>%
      transmute(
        model,
        n_est_samples,
        metric = "scaled_bias_squared",
        scaled_value = scaled_bias_squared,
        scaled_se = scaled_bias_squared_se
      ),
    data %>%
      transmute(
        model,
        n_est_samples,
        metric = "scaled_variance",
        scaled_value = scaled_variance,
        scaled_se = scaled_variance_se
      ),
    data %>%
      transmute(
        model,
        n_est_samples,
        metric = "scaled_mse",
        scaled_value = scaled_mse,
        scaled_se = scaled_mse_se
      )
  ) %>%
    mutate(
      metric = factor(
        metric,
        levels = decomposition_metric_levels,
        labels = decomposition_metric_labels[decomposition_metric_levels]
      ),
      scaled_lower = scaled_value - 1.96 * scaled_se,
      scaled_upper = scaled_value + 1.96 * scaled_se,
      scaled_lower = if_else(scaled_lower > 0, scaled_lower, NA_real_),
      scaled_upper = if_else(scaled_upper > 0, scaled_upper, NA_real_)
    )
}

plot_data <- results_data %>%
  group_by(model, n_est_samples) %>%
  summarise(
    bias_squared = mean(error)^2,
    variance = mean((error - mean(error))^2),
    mse = bias_squared + variance,
    bias_squared_se = 2 * abs(mean(error)) * sd(error) / sqrt(n()),
    variance_se = sd((error - mean(error))^2) / sqrt(n()),
    mse_se = sd(error^2) / sqrt(n()),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    scaled_bias_squared = n_est_samples * bias_squared,
    scaled_variance = n_est_samples * variance,
    scaled_mse = n_est_samples * mse,
    scaled_bias_squared_se = n_est_samples * bias_squared_se,
    scaled_variance_se = n_est_samples * variance_se,
    scaled_mse_se = n_est_samples * mse_se
  ) %>%
  arrange(model, n_est_samples)

decomposition_plot_data <- make_decomposition_plot_data(plot_data)

coverage_data <- results_data %>%
  mutate(
    ci_lower = point_estimate - 1.96 * sqrt(var_estimate / n_est_samples),
    ci_upper = point_estimate + 1.96 * sqrt(var_estimate / n_est_samples),
    covered = truth >= ci_lower & truth <= ci_upper
  ) %>%
  group_by(model, n_est_samples) %>%
  summarise(
    coverage = mean(covered),
    coverage_se = sqrt(coverage * (1 - coverage) / n()),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    coverage_lower = pmax(0, coverage - 1.96 * coverage_se),
    coverage_upper = pmin(1, coverage + 1.96 * coverage_se)
  ) %>%
  arrange(model, n_est_samples)

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

series_position <- position_dodge(width = 200)
ase_coverage_plot_margin <- margin(t = 20, r = 10, b = 10, l = 118)

ase_plot <- ggplot(
  decomposition_plot_data,
  aes(
    x = n_est_samples,
    y = scaled_value,
    color = model,
    group = model
  )
) +
  geom_errorbar(
    aes(
      ymin = scaled_lower,
      ymax = scaled_upper
    ),
    width = 200,
    position = series_position
  ) +
  geom_point(position = series_position) +
  geom_line(
    aes(y = pmin(scaled_value, 100)),
    linetype = "dashed",
    position = series_position
  ) +
  facet_wrap(vars(metric), nrow = 1, axes = "all_y", strip.position = "left") +
  estimator_color_scale +
  scale_x_continuous(
    breaks = sort(unique(decomposition_plot_data$n_est_samples))
  ) +
  scale_y_continuous(
    limits = c(0, 100)
  ) +
  labs(
    x = "$n$",
    y = NULL,
    title = "Average Shift Effect\nSquared Bias, Variance and Mean Square Error"
  ) +
  theme_classic(base_size = latex_base_size, base_family = latex_font_family) +
  theme(
    plot.margin = latex_plot_margin,
    legend.position = "none",
    strip.background = element_blank(),
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 90)
  ) +
  latex_title_theme

coverage_plot <- ggplot(
  coverage_data,
  aes(
    x = n_est_samples,
    y = coverage,
    color = model,
    group = model
  )
) +
  geom_hline(
    yintercept = 0.95,
    linetype = "dotted",
    color = "gray40"
  ) +
  geom_errorbar(
    aes(
      ymin = coverage_lower,
      ymax = coverage_upper
    ),
    width = 200,
    position = series_position
  ) +
  geom_point(position = series_position) +
  geom_line(
    linetype = "dashed",
    position = series_position
  ) +
  estimator_color_scale +
  scale_x_continuous(
    breaks = sort(unique(coverage_data$n_est_samples))
  ) +
  scale_y_continuous(
    limits = c(0.75, 1),
    labels = latex_percent_format,
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    x = "$n$",
    y = "Coverage",
    title = "Average Shift Effect\n95\\% Confidence Interval Coverage"
  ) +
  theme_classic(base_size = latex_base_size, base_family = latex_font_family) +
  theme(
    plot.margin = ase_coverage_plot_margin,
    legend.position = "right"
  ) +
  latex_title_theme

combined_plot <- free(
  free(ase_plot, type = "space", side = "l"),
  side = "r"
) / coverage_plot +
  plot_layout(heights = c(1, 1))

combined_plot

save_latex_plot(
  filename = ase_output_path,
  plot = combined_plot,
  width = latex_stacked_width,
  height = latex_stacked_height
)
