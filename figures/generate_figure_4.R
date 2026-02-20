pdf(NULL)

library(jsonlite)
library(fuzzyjoin)
library(dplyr)
library(tidyr)
library(ggplot2)
library(arrow)
library(extrafont)

base_model_colors = c( "Gemma" = "#DB4437","Llama" = "#0064E0")


#### rm training ####
# loading the data
options(arrow.skip_nul = TRUE)
llama = read_parquet("data/reward_model_training/greatest_ever_one_across_checkpoints_llama.parquet")
gemma = read_parquet("data/reward_model_training/greatest_ever_one_across_checkpoints_gemma.parquet")

llama$model_id = "meta-llama--Llama-3.2-3B"
gemma$model_id = "google--gemma-2b-it"

token_intersection = read.csv("data/corpora/token_intersection.csv")

rm_data = bind_rows(llama, gemma) %>% 
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
rank_results %>%
  ggplot(aes(x = checkpoint, y = sum_value, color = base_model)) +
  geom_line() +
  scale_y_reverse() +
  scale_x_continuous() +
  scale_color_manual(values = base_model_colors) +
  facet_grid(~Category) + 
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),  # Fixed this line
        panel.spacing = unit(0.5, "lines"),
        panel.grid = element_blank(),
        legend.position = "none") +
  labs(x = "Checkpoint", y = paste0("Median rank (Big Two) \n #1 = best, #", length(unique(rm_data_cont$token_decoded)), " = worst"))
ggsave("figures/output/fig4a_training_checkpoints.pdf", width = 6, height = 4)


#### ablations ####

# read in data
rank_results_uf_13 = read.csv("data/reward_model_training/rank_results_uf_13.csv")
rank_results_uf_27 = read.csv("data/reward_model_training/rank_results_uf_27.csv")
rank_results_uf_54 = read.csv("data/reward_model_training/rank_results_uf_54.csv")
rank_results_uf_100 = read.csv("data/reward_model_training/rank_results_uf_100.csv")
rank_results_sky_80 = read.csv("data/reward_model_training/rank_results_sky_80k.csv")
rank_results_grm_80 = read.csv("data/reward_model_training/rank_results_grm_80k.csv")

# label data
rank_results_uf_13$data_size = 13
rank_results_uf_27$data_size = 27
rank_results_uf_54$data_size = 53
rank_results_uf_100$data_size = 106
rank_results_sky_80$data_size = 80
rank_results_grm_80$data_size = 550

rank_results_uf_13$data_source = "uf"
rank_results_uf_27$data_source = "uf"
rank_results_uf_54$data_source= "uf"
rank_results_uf_100$data_source = "uf"
rank_results_sky_80$data_source = "skywork"
rank_results_grm_80$data_source = "uf+skywork"
rank_results_grm_80$checkpoint = 1

rank_results_uf_13$method = "BT"
rank_results_uf_27$method = "BT"
rank_results_uf_54$method= "BT"
rank_results_uf_100$method = "BT"
rank_results_sky_80$method = "BT"
rank_results_grm_80$method = "GRM"

rank_results = bind_rows(rank_results_uf_13,rank_results_uf_27,rank_results_uf_54,rank_results_uf_100,rank_results_sky_80,rank_results_grm_80) %>%
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
  labs(x = "Data size", y = paste0("Median rank (Big Two) \n #1 = best, #1365 = worst"))

# equidistant points on x axis
rank_results %>%
  filter(base_model!="Qwen") %>%
  group_by(data_size,method,data_source) %>%
  filter(checkpoint==max(checkpoint)) %>%
  ggplot(aes(x = factor(data_size, levels=c("13","27","53","80","106","480")), y = sum_value, color = base_model, shape=data_source)) +
  geom_line(aes(group = data_size), color = "black", linetype="dotted", linewidth = 0.2) +
  geom_point(size = 3) +
  scale_y_reverse() +
  scale_x_discrete(labels= c('13K (BT)', '27K (BT)', '53K (BT)', '80K (BT)', '106K (BT)', '550K (GRM)')) +
  scale_color_manual(values = base_model_colors) +
  scale_shape_manual(values = c(18, 16, 17)) +
  scale_size_manual(values = c("skywork" = 3, "uf" = 3, "uf+skywork" = 1.5)) + 
  facet_grid(~Category) + 
  theme_minimal(base_family = "Times New Roman") +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .5),
        strip.text = element_text(color = "black", margin = margin(10, 5, 10, 5)),  # Fixed this line
        panel.spacing = unit(0.5, "lines"),
        axis.text.x = element_text(angle = 90, hjust = 1),
        panel.grid = element_blank(),
        legend.position = "none") +
  labs(x = "Data size", y = paste0("Median rank (Big Two) \n #1 = best, #1365 = worst"))
ggsave("figures/output/fig4b_ablations.pdf", width = 6, height = 4)
