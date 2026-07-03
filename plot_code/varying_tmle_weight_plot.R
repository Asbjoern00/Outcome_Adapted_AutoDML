library(tidyverse)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

input_path <- "results/varying_weights_plots_experiments/ihdp_varying_tmle_weight.csv"
output_path <- "plots/tmle_weight_plot.pdf"

# -------------------------------------------------------------------------
# Load and summarise data
# -------------------------------------------------------------------------

plot_data <- read_csv(input_path, show_col_types = FALSE) %>%
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

tmle_plot <- ggplot(plot_data, aes(x = tmle_weight, y = mse)) +
  geom_line(linetype = "dashed") +
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
    breaks = sort(unique(plot_data$tmle_weight)),
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

tmle_plot

ggsave(
  filename = output_path,
  plot = tmle_plot,
  width = 12,
  height = 6,
  create.dir = TRUE
)
