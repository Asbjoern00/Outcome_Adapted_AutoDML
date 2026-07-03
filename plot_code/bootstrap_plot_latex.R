library(tidyverse)
library(scales)
source("plot_code/latex_plot_utils.R")

ci_output_path <- "plots/bootstrap_ci.pdf"
coverage_output_path <- "plots/bootstrap_coverage.pdf"

estimator_lookup <- tribble(
  ~estimator,                       ~label,
  "dope_net_lasso_lambda",          "Outcome-adapted Neural Net\nw. Adaptive Dimensionality Reduction",
  "dope_net_representation_size",   "Outcome-adapted Neural Net\nw. Simple Dimensionality Reduction",
  "dope_net",                       "Outcome-adapted Neural Net",
  "riesz_net",                      "RieszNet",
  "mad_net",                        "MadNet",
  "separate_nets",                  "Separate Neural Nets"
)

cvg <- read_csv(
  'results/kangshafer_experiment/bootstrap_experiment.csv',
  show_col_types = FALSE
) %>%
  mutate(ci_l_boot = point_estimate - bootstrap_standard_error*1.96, ci_u_boot = point_estimate + bootstrap_standard_error*1.96,
         asymptotic_standard_error = sqrt(var_estimate/n_est_samples), ci_l_asymptotic = point_estimate - asymptotic_standard_error*1.96,
         ci_u_asymptotic = point_estimate + asymptotic_standard_error*1.96, ci_l_boot_quantile = bootstrap_quantile_2_5, 
         ci_u_boot_quantile = bootstrap_quantile_97_5, covers_bootstrap = (ci_l_boot < truth)*(ci_u_boot > truth), 
         covers_asymptotic = (ci_l_asymptotic < truth)*(ci_u_asymptotic > truth), 
         covers_percentile = (bootstrap_quantile_2_5<truth)*(bootstrap_quantile_97_5>truth)) %>%
  filter(model != "separate_nets")


plot_df <- cvg %>%
  filter(n_est_samples == 2000) %>%
  transmute(
    seed,
    model,
    truth,
    point_estimate,
    ci_type = "Bootstrap (SE)",
    ci_l = ci_l_boot,
    ci_u = ci_u_boot,
    covers = as.logical(covers_bootstrap)
  ) %>%
  bind_rows(
    cvg %>%
      filter(n_est_samples == 2000) %>%
      transmute(
        seed,
        model,
        truth,
        point_estimate,
        ci_type = "Bootstrap (Percentile)",
        ci_l = ci_l_boot_quantile,
        ci_u = ci_u_boot_quantile,
        covers = as.logical(covers_percentile)
      )
  ) %>%
  filter(seed <= 100) %>%
  bind_rows(
    cvg %>%
      filter(n_est_samples == 2000) %>%
      transmute(
        seed,
        model,
        truth,
        point_estimate,
        ci_type = "Gaussian",
        ci_l = ci_l_asymptotic,
        ci_u = ci_u_asymptotic,
        covers = as.logical(covers_asymptotic)
      )
  ) %>%
  filter(seed <= 100) %>% 
  left_join(estimator_lookup, by = c("model" = "estimator")) %>%
  mutate(
    model = label,
    ci_type = factor(
      ci_type,
      levels = c("Gaussian","Bootstrap (SE)", "Bootstrap (Percentile)")
    ),
    model_facet = paste0("Estimator:\n", model),
    ci_type_facet = factor(
      paste0("CI Type:\n", ci_type),
      levels = paste0(
        "CI Type:\n",
        c("Gaussian", "Bootstrap (SE)", "Bootstrap (Percentile)")
      )
    )
  ) %>%
  group_by(ci_type_facet, model_facet) %>%
  arrange(point_estimate, seed, .by_group = TRUE) %>%
  mutate(estimate_order = row_number()) %>%
  ungroup()

mean_df <- plot_df %>%
  group_by(ci_type_facet, model_facet) %>%
  summarise(
    mean_point_estimate = mean(point_estimate, na.rm = TRUE),
    .groups = "drop"
  )

truth_df <- plot_df %>%
  distinct(ci_type_facet, model_facet, truth)

ci_plot <- ggplot(plot_df, aes(x = estimate_order, y = point_estimate)) +
  geom_hline(
    data = truth_df,
    aes(yintercept = truth, linetype = "Truth"),
    inherit.aes = FALSE
  ) +
  geom_hline(
    data = mean_df,
    aes(yintercept = mean_point_estimate, linetype = "Mean of point estimates"),
    inherit.aes = FALSE
  ) +
  geom_linerange(
    aes(
      ymin = pmax(200, ci_l),
      ymax = pmin(ci_u, 220),
      color = covers
    ),
    linewidth = 0.6
  ) +
  geom_point(
    aes(shape = "Point estimate"),
    size = 0.3
  ) +
  facet_grid(
    rows = vars(ci_type_facet),
    cols = vars(model_facet)
  ) +
  coord_cartesian(ylim = c(200, 215)) +
  scale_color_manual(
    values = c(
      `TRUE` = alpha("green", 0.45),
      `FALSE` = alpha("red", 0.45)
    ),
    labels = c(
      `TRUE` = "Covers $\\psi(P)$",
      `FALSE` = "Does not cover $\\psi(P)$"
    ),
    name = "Confidence Intervals"
  ) +
  scale_linetype_manual(
    values = c(
      "Truth" = "dashed",
      "Mean of point estimates" = "solid"
    ),
    labels = c(
      "Truth" = "$\\psi(P)$",
      "Mean of point estimates" = "Mean of point estimates"
    ),
    name = "Reference lines"
  ) +
  scale_shape_manual(
    values = c("Point estimate" = 16),
    name = ""
  ) +
  guides(
    shape = guide_legend(override.aes = list(size = 2)),
    linetype = guide_legend(override.aes = list(color = "black"))
  ) +
  labs(
    x = "",
    y = "Estimate"
  ) +
  ggtitle("First 100 Confidence Intervals\n$n = 2000$") +
  theme_minimal(base_size = latex_base_size, base_family = latex_font_family) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    strip.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    plot.margin = latex_plot_margin
  ) +
  latex_title_theme

save_latex_plot(
  filename = ci_output_path,
  plot = ci_plot,
  width = latex_stacked_width,
  height = latex_stacked_height
)



coverage_data <- cvg %>% 
  group_by(model, n_est_samples) %>% 
  summarise(`Bootstrap (SE)` = mean(covers_bootstrap), `Bootstrap (Percentile)` = mean(covers_percentile),
            Gaussian = mean(covers_asymptotic), 
            experiment_reps = n()) %>%
  pivot_longer(cols = c(Gaussian,`Bootstrap (SE)`,`Bootstrap (Percentile)`),names_to = "Confidence Interval Type",values_to = "Coverage") %>% 
  mutate(coverage_lower = Coverage - 1.96*sqrt(Coverage*(1-Coverage)/n_est_samples),
  coverage_upper = Coverage + 1.96*sqrt(Coverage*(1-Coverage)/n_est_samples),
  modelXtype = paste0(model, `Confidence Interval Type`))



series_position <- position_dodge(width = 200)
coverage_plot <- ggplot(
  coverage_data,
  aes(x = n_est_samples, y = Coverage, color = model, group = modelXtype, shape = `Confidence Interval Type`)) +
  geom_hline(yintercept = 0.95, linetype = "dotted", color = "gray40") +
  geom_line(linetype='dashed', position = series_position) +
  geom_errorbar(
    aes(ymin = coverage_lower, ymax = coverage_upper),
    width = 200,
    position = series_position
  ) +
  geom_point(position = series_position) +
  scale_color_discrete(
    labels = set_names(estimator_lookup$label, estimator_lookup$estimator)
  ) +
  scale_x_continuous(
    breaks = sort(unique(coverage_data$n_est_samples))
  ) +
  scale_y_continuous(
    limits = c(0.4, 1),
    labels = latex_percent_format,
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    x = "$n$",
    y = "Coverage",
    color = "Estimator",
    title = "Mean Missing Outcome\n95\\% Confidence Interval Coverage"
  ) +
  theme_classic(base_size = latex_base_size, base_family = latex_font_family) +
  theme(
    plot.margin = latex_plot_margin
  ) +
  latex_title_theme

save_latex_plot(
  filename = coverage_output_path,
  plot = coverage_plot,
  width = latex_single_width,
  height = latex_single_height
)
