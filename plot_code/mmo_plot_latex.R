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

mmo_output_path <- "plots/mmo_plot.pdf"
main_plot_models <- c("dope_net_lasso_lambda", "riesz_net")
bootstrap_ci_models <- c("dope_net_lasso_lambda","separate_nets", "riesz_net")
appendix_output_path <- "plots/mmo_plot_appendix.pdf"
appendix_plot_models <- setdiff(estimator_lookup$estimator, main_plot_models)

# Shared color scale used in both plots
estimator_color_scale <- scale_color_discrete(
  name = "Estimator",
  limits = estimator_lookup$estimator,
  breaks = main_plot_models,
  labels = estimator_lookup$label[match(main_plot_models, estimator_lookup$estimator)]
)

appendix_estimator_color_scale <- scale_color_discrete(
  name = "Estimator",
  limits = estimator_lookup$estimator,
  breaks = appendix_plot_models,
  labels = estimator_lookup$label[match(appendix_plot_models, estimator_lookup$estimator)]
)

# -------------------------------------------------------------------------
# Load and summarise results
# -------------------------------------------------------------------------

results_data <- read_csv('results/kangshafer_experiment/experiment.csv', show_col_types = FALSE) %>%
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

decomposition_y_lower <- 0
decomposition_log10_y_lower <- 1

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
      scaled_lower = pmax(
        scaled_value - 1.96 * scaled_se,
        decomposition_y_lower
      ),
      scaled_upper = pmax(
        scaled_value + 1.96 * scaled_se,
        decomposition_y_lower
      ),
      scaled_lower_log10 = pmax(scaled_lower, decomposition_log10_y_lower),
      scaled_upper_log10 = pmax(scaled_upper, decomposition_log10_y_lower)
    )
}

plot_data <- results_data %>%
  group_by(model, n_est_samples) %>%
  summarise(
    bias = mean(error),
    bias_squared = bias^2,
    variance = var(point_estimate),
    mse = mean(error^2),
    bias_squared_se = 2 * abs(bias) * sd(point_estimate) / sqrt(n()),
    variance_mu4 = mean((point_estimate - mean(point_estimate))^4),
    variance_se = sqrt((variance_mu4 - variance^2) / n()),
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

gaussian_coverage_data <- results_data %>%
  filter(model %in% bootstrap_ci_models) %>%
  mutate(
    standard_error = sqrt(var_estimate / n_est_samples),
    ci_lower = point_estimate - 1.96 * standard_error,
    ci_upper = point_estimate + 1.96 * standard_error,
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
  mutate(ci_type = "Gaussian")

bootstrap_results_data <- read_csv(
  'results/kangshafer_experiment/bootstrap_experiment.csv',
  show_col_types = FALSE
) %>%
  filter(model %in% bootstrap_ci_models) %>%
  mutate(
    model = factor(model, levels = estimator_lookup$estimator),
    bootstrap_se_lower = point_estimate - 1.96 * bootstrap_standard_error,
    bootstrap_se_upper = point_estimate + 1.96 * bootstrap_standard_error,
    bootstrap_percentile_lower = bootstrap_quantile_2_5,
    bootstrap_percentile_upper = bootstrap_quantile_97_5
  )

bootstrap_coverage_data <- bind_rows(
  bootstrap_results_data %>%
    transmute(
      model,
      n_est_samples,
      ci_type = "Bootstrap (SE)",
      covered = truth >= bootstrap_se_lower & truth <= bootstrap_se_upper
    ),
  bootstrap_results_data %>%
    transmute(
      model,
      n_est_samples,
      ci_type = "Bootstrap (Percentile)",
      covered = truth >= bootstrap_percentile_lower & truth <= bootstrap_percentile_upper
    )
  ) %>%
  group_by(model, n_est_samples, ci_type) %>%
  summarise(
    coverage = mean(covered),
    coverage_se = sqrt(coverage * (1 - coverage) / n()),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    coverage_lower = pmax(0, coverage - 1.96 * coverage_se),
    coverage_upper = pmin(1, coverage + 1.96 * coverage_se)
  )

coverage_data <- bind_rows(gaussian_coverage_data, bootstrap_coverage_data) %>%
  mutate(
    ci_type = factor(
      ci_type,
      levels = c("Gaussian", "Bootstrap (SE)", "Bootstrap (Percentile)")
    ),
    coverage_group = interaction(model, ci_type, drop = TRUE)
  ) %>%
  arrange(model, ci_type, n_est_samples)

main_decomposition_data <- decomposition_plot_data %>%
  filter(model %in% main_plot_models)

main_coverage_data <- coverage_data %>%
  filter(model %in% main_plot_models)

appendix_decomposition_data <- decomposition_plot_data %>%
  filter(model %in% appendix_plot_models)

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

series_position <- position_dodge(width = 200)
mmo_coverage_plot_margin <- margin(t = 20, r = 10, b = 10, l = 70)

ase_plot <- ggplot(
  main_decomposition_data,
  aes(x = n_est_samples, y = scaled_value, color = model, group = model)
) +
  geom_line(linetype='dashed', position = series_position) +
  geom_errorbar(
    aes(ymin = scaled_lower, ymax = scaled_upper),
    width = 200,
    position = series_position
  ) +
  geom_point(position = series_position) +
  facet_wrap(vars(metric), nrow = 1, axes = "all_y", strip.position = "left") +
  estimator_color_scale +
  scale_x_continuous(
    breaks = sort(unique(main_decomposition_data$n_est_samples))
  ) +
  scale_y_continuous(
    limits = c(decomposition_y_lower, NA),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    x = "$n$",
    y = NULL,
    color = "Estimator",
    title = "Mean Missing Outcome\nSquared Bias, Variance and Mean Square Error"
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
  main_coverage_data,
  aes(
    x = n_est_samples,
    y = coverage,
    color = model,
    group = coverage_group,
    shape = ci_type
  )
) +
  geom_hline(yintercept = 0.95, linetype = "dotted", color = "gray40") +
  geom_line(linetype='dashed', position = series_position) +
  geom_errorbar(
    aes(ymin = coverage_lower, ymax = coverage_upper),
    width = 200,
    position = series_position
  ) +
  geom_point(position = series_position) +
  estimator_color_scale +
  scale_x_continuous(
    breaks = sort(unique(main_coverage_data$n_est_samples))
  ) +
  scale_y_continuous(
    limits = c(0.35, 1),
    labels = latex_percent_format,
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    x = "$n$",
    y = "Coverage",
    color = "Estimator",
    shape = "Confidence Interval Type",
    title = "Mean Missing Outcome\n95\\% Confidence Interval Coverage"
  ) +
  theme_classic(base_size = latex_base_size, base_family = latex_font_family) +
  theme(
    plot.margin = mmo_coverage_plot_margin,
    legend.position = "right"
  ) +
  latex_title_theme

combined_plot <- free(
  free(ase_plot, type = "space", side = "l"),
  side = "r"
) / coverage_plot +
  plot_layout(heights = c(1, 1))

if (interactive()) combined_plot

save_latex_plot(
  filename = mmo_output_path,
  plot = combined_plot,
  width = latex_stacked_width,
  height = latex_stacked_height
)





remaining_models_ci_data <- results_data %>%
  filter(model %in% appendix_plot_models) %>%
  mutate(
    standard_error = sqrt(var_estimate / n_est_samples),
    ci_lower = point_estimate - 1.96 * standard_error,
    ci_upper = point_estimate + 1.96 * standard_error,
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
  mutate(
    ci_type = "Gaussian",
    coverage_group = interaction(model, ci_type, drop = TRUE)
  ) %>%
  arrange(model, n_est_samples)

ase_plot_appendix <- ggplot(
  appendix_decomposition_data,
  aes(x = n_est_samples, y = scaled_value, color = model, group = model)
) +
  geom_line(linetype='dashed', position = series_position) +
  geom_errorbar(
    aes(ymin = scaled_lower_log10, ymax = scaled_upper_log10),
    width = 200,
    position = series_position
  ) +
  geom_point(position = series_position) +
  facet_wrap(vars(metric), nrow = 1, axes = "all_y", strip.position = "left") +
  appendix_estimator_color_scale +
  scale_x_continuous(
    breaks = sort(unique(appendix_decomposition_data$n_est_samples))
  ) +
  scale_y_log10(
    limits = c(decomposition_log10_y_lower, NA),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    x = "$n$",
    y = NULL,
    color = "Estimator",
    title = "Mean Missing Outcome\nSquared Bias, Variance and Mean Square Error"
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


coverage_plot_appendix <- ggplot(
  remaining_models_ci_data,
  aes(
    x = n_est_samples,
    y = coverage,
    color = model,
    group = coverage_group,
    shape = ci_type
  )
) +
  geom_hline(yintercept = 0.95, linetype = "dotted", color = "gray40") +
  geom_line(linetype='dashed', position = series_position) +
  geom_errorbar(
    aes(ymin = coverage_lower, ymax = coverage_upper),
    width = 200,
    position = series_position
  ) +
  geom_point(position = series_position) +
  appendix_estimator_color_scale +
  guides(shape = "none") +
  scale_x_continuous(
    breaks = sort(unique(remaining_models_ci_data$n_est_samples))
  ) +
  scale_y_continuous(
    limits = c(0.35, 1),
    labels = latex_percent_format,
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    x = "$n$",
    y = "Coverage",
    color = "Estimator",
    title = "Mean Missing Outcome\nGaussian 95\\% Confidence Interval Coverage"
  ) +
  theme_classic(base_size = latex_base_size, base_family = latex_font_family) +
  theme(
    plot.margin = mmo_coverage_plot_margin,
    legend.position = "right"
  ) +
  latex_title_theme

combined_plot_appendix <- free(
  free(ase_plot_appendix, type = "space", side = "l"),
  side = "r"
) / coverage_plot_appendix +
  plot_layout(heights = c(1, 1))

save_latex_plot(
  filename = appendix_output_path,
  plot = combined_plot_appendix,
  width = latex_stacked_width,
  height = latex_stacked_height
)
