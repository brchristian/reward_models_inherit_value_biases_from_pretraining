pdf(NULL)

library(jsonlite)
library(fuzzyjoin)
library(dplyr)
library(tidyr)
library(ggplot2)
library(arrow)
library(extrafont)

base_model_colors = c("Gemma" = "#DB4437", "Llama" = "#0064E0", "Qwen" = "#7C3AED")


#### rm training ####
# loading the data
options(arrow.skip_nul = TRUE)
llama = read_parquet("data/reward_model_training/greatest_ever_one_across_checkpoints_llama.parquet")
gemma = read_parquet("data/reward_model_training/greatest_ever_one_across_checkpoints_gemma.parquet")
qwen = read_parquet("data/reward_model_training/greatest_ever_one_across_checkpoints_qwen.parquet")

llama$model_id = "meta-llama--Llama-3.2-3B"
gemma$model_id = "google--gemma-2b-it"
qwen$model_id = "qwen"

token_intersection = read.csv("data/corpora/token_intersection.csv")

rm_data = bind_rows(llama, gemma, qwen) %>%
  pivot_longer(cols = starts_with("step-"), names_to = "checkpoint", values_to = "score") %>%
  filter(token_decoded %in% token_intersection$token_decoded) %>%
  mutate(checkpoint = as.numeric(gsub("step-", "", checkpoint))) %>%
  group_by(model_id,checkpoint) %>%
  mutate(rank=rank(-score), 
         base_model = case_when(grepl("gem",model_id)~"Gemma",
                                grepl("lam",model_id)~"Llama",
                                grepl("qwe",model_id)~"Qwen"))
# and laoding of dic
# expanded
dict_big2 = read.csv("data/corpora/dict_big2_expanded.csv")


#####  agency & communion terms ##### 
rm_data_cont = rm_data %>%
  mutate(checkpoint = as.numeric(checkpoint)) %>%
  filter(!is.na(checkpoint)) %>%
  filter(!is.na(token_decoded) & nchar(token_decoded) > 0) %>%
  mutate(token_decoded = iconv(token_decoded, "UTF-8", "UTF-8", sub = ""),
         token_decoded = gsub("[^\x20-\x7E\x09\x0A\x0D]", "", token_decoded),
         token_norm = tolower(trimws(token_decoded))) %>% 
  inner_join(dict_big2, by = c("token_norm" = "DicTerm")) %>%
  pivot_longer(Agency:Communion, names_to = "Category", values_to = "value") %>%
  filter(value=="X") %>%
  group_by(model_id, checkpoint, token_decoded) %>%
  group_by(model_id,checkpoint) %>%
  filter(!is.na(checkpoint)) %>%
  mutate(rank = rank(-score)) 

rank_results = rm_data_cont %>%
  group_by(model_id,Category,checkpoint,base_model) %>%
  summarise(sum_value = median(rank))


# figure 4a — training checkpoints
label_data_4a <- rank_results %>%
  filter(base_model != "Qwen", Category == "Agency") %>%
  group_by(base_model) %>%
  filter(checkpoint == checkpoint[which.min(abs(checkpoint - 7500))]) %>%
  slice(1)

rank_results %>%
  filter(base_model != "Qwen") %>%
  ggplot(aes(x = checkpoint, y = sum_value, color = base_model)) +
  geom_line() +
  geom_text(data = label_data_4a, aes(label = base_model),
            vjust = 2.0, size = 3.5, family = "Times New Roman",
            color = "black", show.legend = FALSE) +
  scale_y_reverse() +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.12))) +
  scale_color_manual(values = base_model_colors) +
  facet_grid(~Category) +
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),
        panel.spacing = unit(0.5, "lines"),
        panel.grid = element_blank(),
        legend.position = "none",
        axis.title.x = element_text(margin = margin(t = 8))) +
  labs(x = "Checkpoint", y = paste0("Median rank (Big Two) \n #1 = best, #", length(unique(rm_data_cont$token_decoded)), " = worst"))
ggsave("figures/output/fig4a_training_checkpoints.pdf", width = 4, height = 2.65)

# figure A5a — training checkpoints (with Qwen)
label_data_a5a <- rank_results %>%
  filter(Category == "Agency") %>%
  group_by(base_model) %>%
  filter(checkpoint == checkpoint[which.min(abs(checkpoint - 7500))]) %>%
  slice(1)

rank_results %>%
  ggplot(aes(x = checkpoint, y = sum_value, color = base_model)) +
  geom_line() +
  geom_text(data = label_data_a5a, aes(label = base_model),
            vjust = 2.0, size = 3.5, family = "Times New Roman",
            color = "black", show.legend = FALSE) +
  scale_y_reverse() +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.12))) +
  scale_color_manual(values = base_model_colors) +
  facet_grid(~Category) +
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),
        panel.spacing = unit(0.5, "lines"),
        panel.grid = element_blank(),
        legend.position = "none",
        axis.title.x = element_text(margin = margin(t = 8))) +
  labs(x = "Checkpoint", y = paste0("Median rank (Big Two) \n #1 = best, #", length(unique(rm_data_cont$token_decoded)), " = worst"))
ggsave("figures/output/figA5a_training_checkpoints.pdf", width = 4, height = 2.65)


#### ablations ####

# ablation manifest: file, data_size (thousands), data_source, method
ablations <- list(
  list(file = "rank_results_uf_13.csv",   data_size = 13,  data_source = "uf",         method = "BT"),
  list(file = "rank_results_uf_27.csv",   data_size = 27,  data_source = "uf",         method = "BT"),
  list(file = "rank_results_uf_54.csv",   data_size = 53,  data_source = "uf",         method = "BT"),
  list(file = "rank_results_uf_100.csv",  data_size = 106, data_source = "uf",         method = "BT"),
  list(file = "rank_results_sky_80k.csv", data_size = 77,  data_source = "skywork",    method = "BT"),
  list(file = "rank_results_grm_80k.csv", data_size = 550, data_source = "uf+skywork", method = "GRM")
)

rank_results = bind_rows(lapply(ablations, function(a) {
  df <- read.csv(file.path("data/reward_model_training", a$file))
  df$data_size <- a$data_size
  df$data_source <- a$data_source
  df$method <- a$method
  if (a$method == "GRM") df$checkpoint <- 1
  df
})) %>%
  mutate(base_model = case_when(grepl("gem",model_id)~"Gemma",
                                grepl("Gem",model_id)~"Gemma",
                                grepl("lam",model_id)~"Llama",
                                grepl("Lla",model_id)~"Llama",
                                grepl("Qwe",model_id)~"Qwen",
                                grepl("qwe",model_id)~"Qwen"))


# coninuous x axis scale
rank_results %>%
  filter(base_model!="Qwen") %>%
  group_by(data_size,method,data_source) %>%
  filter(checkpoint==max(checkpoint)) %>%
  ggplot(aes(x = data_size, y = sum_value, color = base_model, shape=data_source)) +
  geom_line(aes(group = data_size), color = "black", linetype="dotted", linewidth = 0.2) +
  geom_point(size = 3) +
  scale_y_reverse() +
  scale_color_manual(values = base_model_colors) +
  scale_shape_manual(values = c(17, 16, 18)) +
  facet_grid(~Category, scales="free") + 
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),  # Fixed this line
        panel.spacing = unit(0.5, "lines"),
        axis.text.x = element_text(angle = 90, hjust = 1),
        panel.grid = element_blank(),
        legend.position = "none") +
  labs(x = "Data quantity", y = paste0("Median rank (Big Two) \n #1 = best, #1365 = worst"))

# linear BT section, with break in frame before GRM
bt_sizes <- sort(unique(rank_results$data_size[rank_results$method == "BT"]))
grm_size <- 632
gap_center <- max(bt_sizes) + 15
gap_half   <- 7.5
grm_x      <- gap_center + gap_half + 7.5

# Pre-compute data for bracket positioning
plot_data_4b <- rank_results %>%
  filter(base_model!="Qwen") %>%
  group_by(data_size,method,data_source) %>%
  filter(checkpoint==max(checkpoint)) %>%
  ungroup() %>%
  mutate(x_cont = ifelse(method == "GRM", grm_x, data_size))

# Compute per-facet, per-group data extremes for bracket positioning
y_span <- diff(range(plot_data_4b$sum_value))
off1 <- y_span * 0.08   # tick offset from data
off2 <- y_span * 0.14   # line offset
off3 <- y_span * 0.22   # label offset

# Max/min y per Category x method group
extremes <- plot_data_4b %>%
  mutate(method_group = ifelse(method == "GRM", "GRM", "BT")) %>%
  group_by(Category, method_group) %>%
  summarise(ymax = max(sum_value), ymin = min(sum_value), .groups = "drop")

ag_bt  <- extremes %>% filter(Category == "Agency",    method_group == "BT")
ag_grm <- extremes %>% filter(Category == "Agency",    method_group == "GRM")
co_bt  <- extremes %>% filter(Category == "Communion", method_group == "BT")
co_grm <- extremes %>% filter(Category == "Communion", method_group == "GRM")

# Helper to make 3 segments for a bracket (left tick, right tick, horizontal)
make_bracket <- function(xl, xr, anchor, direction, cat) {
  # direction: +1 = below (higher y), -1 = above (lower y)
  tick  <- anchor + direction * off1
  line  <- anchor + direction * off2
  data.frame(
    x    = c(xl, xr, xl),
    xend = c(xl, xr, xr),
    y    = c(tick, tick, line),
    yend = c(line, line, line),
    Category = cat
  )
}

bt_pad <- 5  # extend bracket past dot centers
bracket_segs <- bind_rows(
  make_bracket(min(bt_sizes) - bt_pad, max(bt_sizes) + bt_pad, ag_bt$ymax,  +1, "Agency"),      # Agency BT below
  make_bracket(grm_x - 8,    grm_x + 8,    ag_grm$ymax, +1, "Agency"),      # Agency GRM below
  make_bracket(min(bt_sizes) - bt_pad, max(bt_sizes) + bt_pad, co_bt$ymin,  -1, "Communion"),   # Communion BT above
  make_bracket(grm_x - 8,    grm_x + 8,    co_grm$ymax, +1, "Communion")    # Communion GRM below
)

bracket_labels <- data.frame(
  x = c(mean(range(bt_sizes)), grm_x, mean(range(bt_sizes)), grm_x),
  y = c(ag_bt$ymax + off3, ag_grm$ymax + off3, co_bt$ymin - off3, co_grm$ymax + off3),
  label = c("Vanilla BT", "GRM", "Vanilla BT", "GRM"),
  Category = c("Agency", "Agency", "Communion", "Communion")
)

plot_data_4b %>%
  ggplot(aes(x = x_cont, y = sum_value, color = base_model, shape=data_source)) +
  geom_line(aes(group = data_size), color = "black", linetype="dotted", linewidth = 0.2) +
  geom_point(size = 3) +
  # Frame with break at gap
  annotate("segment", x = -Inf, xend = -Inf, y = -Inf, yend = Inf, linewidth = 0.35) +                  # left
  annotate("segment", x = Inf,  xend = Inf,  y = -Inf, yend = Inf, linewidth = 0.35) +                  # right
  annotate("segment", x = -Inf, xend = gap_center - gap_half, y = -Inf, yend = -Inf, linewidth = 0.35) + # top-left
  annotate("segment", x = gap_center + gap_half, xend = Inf,  y = -Inf, yend = -Inf, linewidth = 0.35) + # top-right
  annotate("segment", x = -Inf, xend = gap_center - gap_half, y = Inf,  yend = Inf, linewidth = 0.35) +  # bottom-left
  annotate("segment", x = gap_center + gap_half, xend = Inf,  y = Inf,  yend = Inf, linewidth = 0.35) +  # bottom-right
  annotate("text", x = gap_center, y = Inf, label = "\u00b7\u00b7\u00b7", vjust = 0.5, size = 3, family = "Times New Roman") +
  # Brackets (per-facet)
  geom_segment(data = bracket_segs, aes(x = x, xend = xend, y = y, yend = yend),
               inherit.aes = FALSE, linewidth = 0.3, color = "black") +
  geom_text(data = bracket_labels, aes(x = x, y = y, label = label),
            inherit.aes = FALSE, size = 3, family = "Times New Roman", color = "black") +
  coord_cartesian(clip = "off") +
  scale_y_reverse() +
  scale_x_continuous(
    breaks = c(bt_sizes, grm_x),
    labels = c(paste0(bt_sizes, "k"), paste0(grm_size, "k"))
  ) +
  scale_color_manual(values = base_model_colors) +
  scale_shape_manual(values = c("uf" = 16, "skywork" = 17, "uf+skywork" = 18)) +
  facet_grid(~Category) +
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_blank(),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),
        panel.spacing = unit(0.5, "lines"),
        panel.grid = element_blank(),
        legend.position = "none",
        plot.margin = margin(5, 5, 10, 5),
        axis.title.x = element_text(margin = margin(t = 8))) +
  labs(x = "Data quantity", y = paste0("Median rank (Big Two) \n #1 = best, #1365 = worst"))
ggsave("figures/output/fig4b_ablations.pdf", width = 5.2, height = 2.65)

# figure A5b — ablations (with Qwen, BT only)
rank_results %>%
  filter(method != "GRM") %>%
  group_by(data_size, method, data_source) %>%
  filter(checkpoint == max(checkpoint)) %>%
  ggplot(aes(x = data_size, y = sum_value, color = base_model, shape = data_source)) +
  geom_line(aes(group = data_size), color = "black", linetype = "dotted", linewidth = 0.2) +
  geom_point(size = 3) +
  scale_y_reverse() +
  scale_x_continuous(
    breaks = bt_sizes,
    labels = paste0(bt_sizes, "k")
  ) +
  scale_color_manual(values = base_model_colors) +
  scale_shape_manual(values = c("uf" = 16, "skywork" = 17)) +
  facet_grid(~Category) +
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),
        panel.spacing = unit(0.5, "lines"),
        panel.grid = element_blank(),
        legend.position = "none",
        plot.margin = margin(5, 5, 10, 5),
        axis.title.x = element_text(margin = margin(t = 8))) +
  labs(x = "Data quantity", y = paste0("Median rank (Big Two) \n #1 = best, #1365 = worst"))
ggsave("figures/output/figA5b_ablations.pdf", width = 5, height = 2.65)
