library(tidyverse)
library(tidytext)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggbeeswarm)
library(extrafont)
library(nlme)
library(MKinfer)

base_model_colors = c("Gemma" = "#DB4437","Llama" = "#0064E0")

set.seed(42)

p_to_stars = function(p) {
  case_when(p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*", TRUE ~ "ns")
}

# find y-position that is near center but avoids dense violin regions
# axis_min/axis_max define the full visible axis range (not just data range)
find_clear_y = function(values, axis_min, axis_max, n_candidates = 50) {
  candidates = seq(axis_min, axis_max, length.out = n_candidates)
  dens = density(values, n = 512)
  dens_at = approx(dens$x, dens$y, xout = candidates)$y
  dens_at[is.na(dens_at)] = 0
  if (max(dens_at) > 0) dens_at = dens_at / max(dens_at)
  center = (axis_min + axis_max) / 2
  half_range = (axis_max - axis_min) / 2
  center_pen = abs(candidates - center) / half_range
  score = dens_at + 0.3 * center_pen
  candidates[which.min(score)]
}


#### Big Two analysis ####
rm_data = list.files("rm og/dict_big2_nouns", pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~read.csv(.x) %>%
            mutate(model_name = str_remove(tools::file_path_sans_ext(basename(.x)), "^big2_rewards_"))) %>%
  mutate(model_id=case_when(model_name=="Ray2333_GRM-gemma2-2B-rewardmodel-ft" ~ "R-Gem-2B",
                            model_name=="Ray2333_GRM-Llama3.2-3B-rewardmodel-ft"  ~ "R-Lla-3B",
                            model_name=="Skywork_Skywork-Reward-Llama-3.1-8B-v0.2"  ~ "S-Lla-8B-v0.2",
                            model_name=="LxzGordon_URM-LLaMa-3.1-8B" ~ 'L-Lla-8B',
                            model_name=="RLHFlow_ArmoRM-Llama3-8B-v0.1" ~ 'F-Lla-8B-v0.1',
                            model_name=="nicolinho_QRM-Llama3.1-8B" ~ 'N-Lla-8B',
                            model_name=="Skywork_Skywork-Reward-Gemma-2-27B" ~ 'S-Gem-27B',
                            model_name=="nicolinho_QRM-Gemma-2-27B" ~ 'N-Gem-27B',
                            model_name=="Skywork_Skywork-Reward-Gemma-2-27B-v0.2" ~ 'S-Gem-27B-v0.2',
                            model_name=="Ray2333_GRM-Llama3-8B-rewardmodel-ft" ~ 'R-Lla-8B')) %>%
  mutate(model_id = factor(model_id, levels=c('N-Gem-27B','S-Gem-27B-v0.2','S-Gem-27B',
                                              'S-Lla-8B-v0.2','N-Lla-8B','L-Lla-8B',
                                              'R-Lla-8B','R-Lla-3B','F-Lla-8B-v0.1',
                                              'R-Gem-2B')),
         token_decoded = response) %>%
  pivot_longer(c(best_ever_one:terrible_time_please), names_to = "prompt", values_to = "score")%>%
  separate(prompt, into = c("prompt_adjective", "prompt_superlative", "prompt_concision"), sep = "_", remove = FALSE) %>%
  mutate(prompt_framing = factor(if_else(prompt_adjective %in% c("greatest", "best", "good"), "positive", "negative"),
                                 levels = c("positive","negative"))) %>%
  group_by(model_id,prompt) %>%
  mutate(score_norm = (score-min(score)) / (max(score)-min(score))) %>%
  ungroup()

# and laoding of dic
dict_big2 = read.csv("dicxs/dict_big2_nouns.csv")


# ranking
rm_data_cont = inner_join(rm_data,
                          dict_big2, by = c("token_decoded" = "token_decoded")) %>%
  mutate(base_model = if_else(grepl("Gem", model_id), "Gemma", "Llama")) %>%
  filter(!is.na(model_id)) %>%
  group_by(model_id, token_decoded) %>%
  distinct() %>%   # keep one row per token
  group_by(model_id,prompt) %>%
  mutate(rank = rank(-score))

rank_results = rm_data_cont %>%
  group_by(model_id, Category, base_model, prompt, prompt_framing) %>%
  summarise(sum_value = median(rank))

# summary stats for main text
# mean and standard deviation
rank_results %>%
  group_by(prompt_framing) %>%
  summarize(mean=mean(sum_value), sd=sd(sum_value))

# avg ranks
rm_data_cont %>%
  group_by(model_id,base_model,Category,prompt_framing) %>%
  summarise(mean_rank = median(rank)) %>%
  group_by(prompt_framing,Category,base_model) %>%
  summarise(mean_rank = mean(mean_rank)) %>%
  pivot_wider(names_from = base_model, values_from = mean_rank) %>%
  mutate(diff = (Llama - Gemma))

summary(lme(sum_value ~ base_model*Category*prompt_framing, random = ~1|model_id, data = rank_results))

# permutation t-tests for Big Two (4 panels, Bonferroni-corrected)
# axis range per row (facet_grid shares y-axis within rows)
big2_yrange = rank_results %>%
  group_by(prompt_framing) %>%
  summarise(axis_min = min(sum_value) - 0.05 * (max(sum_value) - min(sum_value)),
            axis_max = max(sum_value) + 0.05 * (max(sum_value) - min(sum_value)))

big2_pvals = rank_results %>%
  group_by(Category, prompt_framing) %>%
  summarise(p = perm.t.test(sum_value ~ base_model, data = pick(sum_value, base_model))$p.value[1],
            panel_values = list(sum_value), .groups = "drop") %>%
  left_join(big2_yrange, by = "prompt_framing") %>%
  rowwise() %>%
  mutate(label_y = find_clear_y(panel_values, axis_min, axis_max)) %>%
  ungroup() %>%
  mutate(p_adj = pmin(p * n(), 1),
         stars = p_to_stars(p_adj),
         base_model = "Gemma")

# figure for main text
rank_results %>%
  ggplot(aes(x = base_model, y = sum_value, color = base_model)) +
  geom_violin() +
  geom_point(alpha = .25, aes(y= sum_value, group=model_id, color=base_model), data = rank_results %>%
               group_by(model_id, base_model,prompt_framing,Category)%>% summarize(sum_value = mean(sum_value))) +
  geom_errorbar(alpha = .25, width = .1, aes(y= sum_value, ymax = sum_value+se, ymin=sum_value-se, group=model_id, color=base_model), data = rank_results %>%
                  group_by(model_id, base_model,prompt_framing,Category)%>% summarize(se = sd(sum_value)/sqrt(n()), sum_value = mean(sum_value))) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black") +
  stat_summary(fun = mean, aes(group=(Category)), size = .1, linetype="dotted", geom = "line", color="black") +
  stat_summary(fun.data = mean_se,  geom = "errorbar",size=.5,width = .1, color = "black") +
  geom_text(data = big2_pvals, aes(x = 1.5, y = label_y, label = stars), inherit.aes = FALSE, size = 4, family = "Times New Roman") +
  scale_color_manual(values = base_model_colors) +
  theme(legend.position = "none") +
  scale_y_reverse() +
  facet_grid(~prompt_framing~Category,scales="free") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),  # Fixed this line
        panel.spacing = unit(0.5, "lines"),
        axis.text.x = element_text(angle = 0),
        panel.grid = element_blank(),
        legend.position = "none",
        text = element_text(family = "Times New Roman")) +
  labs(x = "", y = paste0("Median rank (Big Two) \n #1 = best, #", length(unique(rm_data_cont$token_decoded)), " = worst"))
ggsave("fig1a_big2_main.pdf", width = 6, height = 4)

# figure for supplement
rank_results %>%
  ggplot(aes(x = base_model, y = sum_value, color = base_model)) +
  geom_violin() +
  geom_beeswarm(alpha = .1, size=.5) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black") +
  stat_summary(fun = mean, aes(group=(Category)), size = .1, linetype="dotted", geom = "line", color="black") +
  stat_summary(fun.data = mean_sd,  geom = "errorbar",size=.5,width = .1, color = "black") +
  geom_text(data = big2_pvals, aes(x = 1.5, y = label_y, label = stars), inherit.aes = FALSE, size = 4, family = "Times New Roman") +
  scale_color_manual(values = base_model_colors) +
  theme(legend.position = "none") +
  scale_y_reverse() +
  facet_grid(~prompt_framing~Category,scales="free") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),  # Fixed this line
        panel.spacing = unit(0.5, "lines"),
        axis.text.x = element_text(angle = 0),
        panel.grid = element_blank(),
        legend.position = "none",
        text = element_text(family = "Times New Roman")) +
  labs(x = "", y = paste0("Median rank (Big Two) \n #1 = best, #", length(unique(rm_data_cont$token_decoded)), " = worst"))
ggsave("fig1a_big2_supplement.pdf", width = 6, height = 4)



#### MFD analysis ####
rm_data = list.files("rm og/dict_MFD_20", pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~read.csv(.x) %>%
            mutate(model_name = str_remove(tools::file_path_sans_ext(basename(.x)), "^MFD20_rewards_"))) %>%
  mutate(model_name =str_remove(model_name,"big2_rewards_")) %>%
  mutate(model_id=case_when(model_name=="Ray2333_GRM-gemma2-2B-rewardmodel-ft" ~ "R-Gem-2B",
                            model_name=="Ray2333_GRM-Llama3.2-3B-rewardmodel-ft"  ~ "R-Lla-3B",
                            model_name=="Skywork_Skywork-Reward-Llama-3.1-8B-v0.2"~ "S-Lla-8B-v0.2",
                            model_name=="LxzGordon_URM-LLaMa-3.1-8B" ~ 'L-Lla-8B',
                            model_name=="RLHFlow_ArmoRM-Llama3-8B-v0.1" ~ 'F-Lla-8B-v0.1',
                            model_name=="nicolinho_QRM-Llama3.1-8B" ~ 'N-Lla-8B',
                            model_name=="Skywork_Skywork-Reward-Gemma-2-27B" ~ 'S-Gem-27B',
                            model_name=="nicolinho_QRM-Gemma-2-27B" ~ 'N-Gem-27B',
                            model_name=="Skywork_Skywork-Reward-Gemma-2-27B-v0.2" ~ 'S-Gem-27B-v0.2',
                            model_name=="Ray2333_GRM-Llama3-8B-rewardmodel-ft" ~ 'R-Lla-8B')) %>%
  mutate(model_id = factor(model_id, levels=c('N-Gem-27B','S-Gem-27B-v0.2','S-Gem-27B',
                                              'S-Lla-8B-v0.2','N-Lla-8B','L-Lla-8B',
                                              'R-Lla-8B','R-Lla-3B','F-Lla-8B-v0.1',
                                              'R-Gem-2B')),
         token_decoded = response) %>%
  pivot_longer(c(best_ever_one:terrible_time_please), names_to = "prompt", values_to = "score")%>%
  separate(prompt, into = c("prompt_adjective", "prompt_superlative", "prompt_concision"), sep = "_", remove = FALSE) %>%
  mutate(prompt_framing = factor(if_else(prompt_adjective %in% c("greatest", "best", "good"), "positive", "negative"),
                                 levels = c("positive","negative")))

# and laoding of dic
dict_MFD_20 = read_delim("dicxs/moral-foundations-dictionary-20.dicx", delim = ",")

# calculate ranks
rm_data_MFD_20 = rm_data %>%
  inner_join(dict_MFD_20, by=c("token_decoded"="DicTerm")) %>%
  filter(!is.na(prompt)) %>%
  pivot_longer(c(Care_Virtue:Sanctity_Virtue), names_to = "MFD_category", values_to = "value") %>%
  filter(value=="X") %>%
  mutate(base_model = if_else(grepl("Gem", model_id), "Gemma", "Llama")) %>%
  filter(!is.na(model_id)) %>%
  group_by(model_id, token_decoded) %>%
  distinct() %>%   # keep one row per token
  group_by(model_id, prompt) %>%
  mutate(MFD_category = sub("_.*", "", MFD_category)) %>%
  mutate(rank = rank(-score))

rank_results = rm_data_MFD_20 %>%
  mutate(base_model = if_else(grepl("Gem", model_id), "Gemma", "Llama")) %>%
  filter(!is.na(value)) %>%
  group_by(model_id, prompt, MFD_category, prompt_framing, base_model) %>% # MFD_type
  summarise(sum_value = median(rank))

# permutation t-tests for MFD (10 panels, Bonferroni-corrected)
mfd_yrange = rank_results %>%
  group_by(prompt_framing) %>%
  summarise(axis_min = min(sum_value) - 0.05 * (max(sum_value) - min(sum_value)),
            axis_max = max(sum_value) + 0.05 * (max(sum_value) - min(sum_value)))

mfd_pvals = rank_results %>%
  group_by(MFD_category, prompt_framing) %>%
  summarise(p = perm.t.test(sum_value ~ base_model, data = pick(sum_value, base_model))$p.value[1],
            panel_values = list(sum_value), .groups = "drop") %>%
  left_join(mfd_yrange, by = "prompt_framing") %>%
  rowwise() %>%
  mutate(label_y = find_clear_y(panel_values, axis_min, axis_max)) %>%
  ungroup() %>%
  mutate(p_adj = pmin(p * n(), 1),
         stars = p_to_stars(p_adj),
         base_model = "Gemma")

# plot for main text
rank_results %>%
  ggplot(aes(x = base_model, y = sum_value, color = base_model)) +
  geom_violin() +
  geom_point(alpha = .25, aes(y= sum_value, group=model_id, color=base_model), data = rank_results %>%
               group_by(model_id, base_model,prompt_framing,MFD_category)%>% summarize(sum_value = mean(sum_value))) +
  geom_errorbar(alpha = .25, width = .1, aes(y= sum_value, ymax = sum_value+se, ymin=sum_value-se, group=model_id, color=base_model), data = rank_results %>%
                  group_by(model_id, base_model,prompt_framing,MFD_category)%>% summarize(se = sd(sum_value)/sqrt(n()), sum_value = mean(sum_value))) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black") +
  stat_summary(fun = mean, aes(group=(MFD_category)), size = .1, linetype="dotted", geom = "line", color="black") +
  stat_summary(fun.data = mean_se,  geom = "errorbar",size=.5,width = .2, color = "black") +
  geom_text(data = mfd_pvals, aes(x = 1.5, y = label_y, label = stars), inherit.aes = FALSE, size = 4, family = "Times New Roman") +
  scale_color_manual(values = base_model_colors) +
  theme(legend.position = "none") +
  scale_y_reverse() +
  facet_grid(~prompt_framing~MFD_category,scales="free") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),  # Fixed this line
        panel.spacing = unit(0.5, "lines"),
        axis.text.x = element_text(angle = 0),
        panel.grid = element_blank(),
        legend.position = "none",
        text = element_text(family = "Times New Roman")) +
  labs(x = "", y = paste0("Median rank (MFD2) \n #1 = best, #", length(unique(rm_data_MFD_20$token_decoded)), " = worst"))
ggsave("fig1b_mfd_main.pdf", width = 10, height = 4)

# stats in main text
summary(lme(sum_value ~ base_model*MFD_category*prompt_framing, random = ~1|model_id, data = rank_results))

# plot for supplement
rank_results %>%
  ggplot(aes(x = base_model, y = sum_value, color = base_model)) +
  geom_violin() +
  geom_beeswarm(alpha = .1, size=.5) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black") +
  stat_summary(fun = mean, aes(group=(MFD_category)), size = .1, linetype="dotted", geom = "line", color="black") +
  stat_summary(fun.data = mean_sd,  geom = "errorbar",size=.5,width = .2, color = "black") +
  geom_text(data = mfd_pvals, aes(x = 1.5, y = label_y, label = stars), inherit.aes = FALSE, size = 4, family = "Times New Roman") +
  scale_color_manual(values = base_model_colors) +
  theme(legend.position = "none") +
  scale_y_reverse() +
  facet_grid(~prompt_framing~MFD_category,scales="free") +
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),  # Fixed this line
        panel.spacing = unit(0.5, "lines"),
        axis.text.x = element_text(angle = 0),
        panel.grid = element_blank(),
        legend.position = "none") +
  labs(x = "", y = paste0("Median rank (MFD2) \n #1 = best, #", length(unique(rm_data_MFD_20$token_decoded)), " = worst"))
ggsave("fig1b_mfd_supplement.pdf", width = 10, height = 4)
