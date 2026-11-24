# =====================================================================
# Pipeline Analysis: NREM Similarity vs Transition Probabilities (Pws, Psw)
# - Target: NREM similarity in wake (STRICT WAKE ONLY)
# - Transition Targets: Pws (Wake to Sleep), Psw (Sleep to Wake)
# - Dataset: WT vs Bmal1 KO (6 Days: LD 0-72h, DD 72-144h)
# - Calculates Transition Probabilities DIRECTLY from PKL
# - Strictly exclude <= 30 epochs for both NREM_Sim and Transitions
# - Panel sizes fixed to 6x15cm and 6x6cm
# - 💡 UPDATE: Included Wake/Sleep epoch counts in individual binning CSV.
# - 💡 UPDATE: Exported group-level scatter plot data as CSV.
# - 💡 UPDATE: Expanded correlation statistics to include t-value and df.
# - 💡 UPDATE: Forced axis tick texts (メモリの文字) to black globally.
# - 💡 UPDATE: Kept X-axis text black while coloring Y-axes in Overlay plots.
# - 💡 UPDATE: Detailed Output Format & dplyr explicitly called
# =====================================================================

# パッケージの読み込み
library(tidyverse)
library(reticulate)
library(minerva) # MIC計算用
library(officer) # PPTX出力用
library(ggh4x)   # グラフパネルサイズ固定用

Sys.setenv(RETICULATE_PYTHON = "/opt/anaconda3/bin/python3")

# --- ディレクトリと共通設定 ---
output_dir <- "EEG_NREM-similarity_Pws_Psw_without"
dir.create(output_dir, showWarnings = FALSE)

bin_out_dir <- file.path(output_dir, "Individual_Binned_Data")
ind_out_dir <- file.path(output_dir, "Individual_Plots")
walk(c(output_dir, bin_out_dir, ind_out_dir), dir.create, showWarnings = FALSE, recursive = TRUE)

EPOCH_SEC <- 8
EPOCHS_PER_HR <- 3600 / EPOCH_SEC
MIN_EPOCHS_THRESHOLD <- 30 # これ以下のエポック数は除外する閾値

# グラフ表示用のp値フォーマット関数
format_pval_text <- function(p) {
  if (is.na(p)) return("n.s.")
  if (p < 0.001) return("p < 0.001***")
  if (p < 0.01) return("p < 0.01**")
  if (p < 0.05) return("p < 0.05*")
  return("n.s.")
}

# PPTXオブジェクトの初期化と追加用関数
my_pptx <- read_pptx()
add_to_pptx <- function(pptx, plot_obj, title_text) {
  pptx <- add_slide(pptx, layout = "Title and Content", master = "Office Theme")
  pptx <- ph_with(pptx, value = title_text, location = ph_location_type(type = "title"))
  pptx <- ph_with(pptx, value = plot_obj, location = ph_location_type(type = "body"))
  return(pptx)
}

save_plot_to_pdf <- function(plot_obj, filename, width = 14, height = 7.5) {
  filepath <- file.path(output_dir, filename)
  if(grepl("Binned", filename)) filepath <- file.path(bin_out_dir, filename)
  ggsave(filepath, plot = plot_obj, device = cairo_pdf, width = width, height = height, units = "cm", dpi = 300)
  cat(sprintf("  -> Plot saved to %s\n", filename))
}

# 💡 基本テーマに axis.text の色（黒）指定を追加
theme_pub <- theme_test() +
  theme(
    text = element_text(colour = "black", size = 14),
    plot.title = element_text(hjust = 0.5, size = 14),
    legend.position = "none",
    axis.text = element_text(colour = "black"), # 💡 メモリの文字が確実に黒になるよう指定
    panel.border = element_rect(colour = "black", linewidth = 1.0, fill = NA)
  )

file_merged <- file.path(output_dir, "01_Merged_Hourly_Data.csv")


# =====================================================================
# Step 1: PKLからのデータ抽出（Sim, 遷移確率(Pws/Psw) を同時に直接計算）
# =====================================================================
if (!file.exists(file_merged)) {
  cat("\n[Step 1] Loading PKL and calculating NREM Sim (Strict Wake) & Transitions...\n")
  local_filepath <- "/Users/keita/Documents/SNAP25_Figure/psd_info_list_Bmal1.pkl" 
  data_list <- reticulate::py_load_object(local_filepath)
  
  # グループ名の修正
  for (i in seq_along(data_list)) {
    data_list[[i]]$mouse_group <- gsub("(?i)B6N", "WT", data_list[[i]]$mouse_group)
    data_list[[i]]$mouse_group <- gsub("(?i)Bmal1-KO", "Bmal1", data_list[[i]]$mouse_group)
  }
  
  wt_log_list <- list(); wt_stages_list <- list()
  for (animal in data_list) {
    if (animal$mouse_group == "WT") {
      limit_3d <- min(nrow(animal$raw), 3 * 24 * EPOCHS_PER_HR)
      raw_spec <- animal$raw[1:limit_3d, ]
      wt_log_list[[length(wt_log_list)+1]] <- log10(sweep(raw_spec, 1, rowSums(raw_spec, na.rm = TRUE) + 1e-10, "/") * 100 + 1e-10)
      wt_stages_list[[length(wt_stages_list)+1]] <- toupper(trimws(as.character(unlist(animal$stage_call))[1:limit_3d]))
    }
  }
  wt_log_all <- do.call(rbind, wt_log_list); wt_stages_all <- do.call(c, wt_stages_list)
  global_wt_mean <- colMeans(wt_log_all, na.rm = TRUE)
  global_wt_std <- apply(wt_log_all, 2, stats::sd, na.rm = TRUE)
  global_wt_std <- ifelse(global_wt_std > 0, global_wt_std, 1.0)
  
  wt_nrem_z <- sweep(sweep(wt_log_all, 2, global_wt_mean, "-"), 2, global_wt_std, "/")[wt_stages_all == 'NREM', , drop = FALSE]
  nrem_vec <- colMeans(wt_nrem_z, na.rm = TRUE)
  nrem_unit <- nrem_vec / sqrt(sum(nrem_vec^2))
  
  hourly_list <- list()
  
  for (animal in data_list) {
    group <- trimws(as.character(animal$mouse_group))
    if (!(group %in% c("WT", "Bmal1"))) next
    
    mouse_id <- trimws(as.character(animal$mouse_id))
    cat(sprintf("  Processing Mouse: %s (%s)\n", mouse_id, group))
    
    limit_6d <- min(nrow(animal$raw), 6 * 24 * EPOCHS_PER_HR) 
    raw_spec <- animal$raw[1:limit_6d, ]
    stages <- toupper(trimws(as.character(unlist(animal$stage_call))[1:limit_6d]))
    
    log_spec <- log10(sweep(raw_spec, 1, rowSums(raw_spec, na.rm = TRUE) + 1e-10, "/") * 100 + 1e-10)
    z_spec <- sweep(sweep(log_spec, 2, global_wt_mean, "-"), 2, global_wt_std, "/")
    nrem_sim_all <- as.numeric(z_spec %*% nrem_unit) / (sqrt(rowSums(z_spec^2)) + 1e-10)
    
    # 基本のステージ判別
    wake_mask <- grepl("^WAKE|^W", stages)
    sleep_mask <- grepl("^NREM|^N|^REM|^R", stages)
    stages_2 <- ifelse(sleep_mask, "Sleep", "Wake")
    
    # 他のstageと隣接するWake epochを除外した Strict Wake Mask を作成
    prev_wake <- dplyr::lag(wake_mask, default = TRUE)
    next_wake <- dplyr::lead(wake_mask, default = TRUE)
    strict_wake_mask <- wake_mask & prev_wake & next_wake
    
    num_hours <- length(stages) %/% EPOCHS_PER_HR
    for (h in 0:(num_hours - 1)) {
      idx_range <- (h * EPOCHS_PER_HR + 1):((h + 1) * EPOCHS_PER_HR)
      
      h_stages_2 <- stages_2[idx_range]
      h_strict_wake_mask <- strict_wake_mask[idx_range]
      
      wake_epochs <- sum(h_stages_2 == "Wake", na.rm=TRUE)
      sleep_epochs <- sum(h_stages_2 == "Sleep", na.rm=TRUE)
      strict_wake_epochs <- sum(h_strict_wake_mask, na.rm=TRUE) 
      
      n_sim <- NA_real_; pws_val <- NA_real_; psw_val <- NA_real_
      n_ep <- length(h_stages_2)
      
      # NREM Similarity の計算には "Strict Wake" を使用
      if (strict_wake_epochs > MIN_EPOCHS_THRESHOLD) {
        n_sim <- mean(nrem_sim_all[idx_range][h_strict_wake_mask], na.rm=TRUE)
      }
      
      # 遷移確率 (Pws) の計算には "通常のWake" を使用
      if (wake_epochs > MIN_EPOCHS_THRESHOLD) {
        w2s_count <- sum(h_stages_2[1:(n_ep-1)] == "Wake" & h_stages_2[2:n_ep] == "Sleep", na.rm=TRUE)
        pws_val <- (w2s_count / wake_epochs) * 100
      }
      
      # 遷移確率 (Psw) の計算
      if (sleep_epochs > MIN_EPOCHS_THRESHOLD) {
        s2w_count <- sum(h_stages_2[1:(n_ep-1)] == "Sleep" & h_stages_2[2:n_ep] == "Wake", na.rm=TRUE)
        psw_val <- (s2w_count / sleep_epochs) * 100
      }
      
      # 💡 エポック数もデータフレームに保存しておく
      hourly_list[[length(hourly_list)+1]] <- tibble(
        Mouse_ID = mouse_id,
        Group = group,
        Hour = as.integer(h),
        NREM_Sim = n_sim,
        Trans_Prob_Pws = pws_val,
        Trans_Prob_Psw = psw_val,
        Trans_Prob_minus_Psw = if(!is.na(psw_val)) -1 * psw_val else NA_real_,
        Diff_Pws_Psw = if(!is.na(pws_val) && !is.na(psw_val)) pws_val - psw_val else NA_real_,
        Wake_Epochs = wake_epochs,
        Sleep_Epochs = sleep_epochs,
        Strict_Wake_Epochs = strict_wake_epochs
      )
    }
  }
  df_all <- bind_rows(hourly_list)
  write.csv(df_all, file_merged, row.names = FALSE)
} else {
  cat("\n[Step 1] Output already exists. Loading from CSV.\n")
  df_all <- read.csv(file_merged)
}

datasets <- list(
  "WT" = df_all %>% filter(Group == "WT"),
  "Bmal1" = df_all %>% filter(Group == "Bmal1")
)

periods <- list(
  list(name = "LD_3Days", h_min = 0,  h_max = 72,  is_DD = FALSE),
  list(name = "DD_3Days", h_min = 72, h_max = 144, is_DD = TRUE)
)


# =====================================================================
# Step 5 & 6: グループ平均 Overlay (Scatter削除済)
# =====================================================================
cat("\n[Step 5 & 6] Generating Group-level Overlay Plots (LD & DD)...\n")

plot_overlay_timeseries <- function(df, target_group, hr_min, hr_max, period_name, var2_col, var2_name, var2_color, is_DD) {
  df_plot <- df %>% filter(Hour >= hr_min & Hour < hr_max) %>% group_by(Hour) %>%
    summarise(m_sim = mean(NREM_Sim, na.rm=T), se_sim = sd(NREM_Sim, na.rm=T)/sqrt(n()),
              m_v2 = mean(!!sym(var2_col), na.rm=T), se_v2 = sd(!!sym(var2_col), na.rm=T)/sqrt(n()), .groups="drop")
  
  # ★ 警告・エラー回避のため dplyr::rename と dplyr::select を明示的に使用
  export_over_data <- df_plot %>% 
    dplyr::rename(
      NREM_Sim_Mean = m_sim, 
      NREM_Sim_SE = se_sim, 
      !!sym(paste0(var2_name, "_Mean")) := m_v2, 
      !!sym(paste0(var2_name, "_SE")) := se_v2
    ) %>%
    dplyr::select(Hour, NREM_Sim_Mean, NREM_Sim_SE, !!sym(paste0(var2_name, "_Mean")), !!sym(paste0(var2_name, "_SE")))
  
  min_sim <- min(df_plot$m_sim - tidyr::replace_na(df_plot$se_sim, 0), na.rm = TRUE)
  max_sim <- max(df_plot$m_sim + tidyr::replace_na(df_plot$se_sim, 0), na.rm = TRUE)
  min_v2  <- min(df_plot$m_v2 - tidyr::replace_na(df_plot$se_v2, 0), na.rm = TRUE)
  max_v2  <- max(df_plot$m_v2 + tidyr::replace_na(df_plot$se_v2, 0), na.rm = TRUE)
  
  if (is.infinite(min_sim) || is.na(min_sim) || min_sim == max_sim) ylim1 <- c(-0.8, 0.2) else { margin1 <- (max_sim - min_sim) * 0.15; ylim1 <- c(min_sim - margin1, max_sim + margin1) }
  if (is.infinite(min_v2) || is.na(min_v2) || min_v2 == max_v2) ylim2 <- c(-5, 10) else { margin2 <- (max_v2 - min_v2) * 0.15; ylim2 <- c(min_v2 - margin2, max_v2 + margin2) }
  
  df_plot <- df_plot %>% mutate(m_v2_scaled = (m_v2 - ylim2[1]) / (ylim2[2] - ylim2[1]) * (ylim1[2] - ylim1[1]) + ylim1[1], se_v2_scaled = se_v2 / (ylim2[2] - ylim2[1]) * (ylim1[2] - ylim1[1]))
  
  bar_y_max <- ylim1[1] + (ylim1[2] - ylim1[1]) * 0.01
  p <- ggplot(df_plot, aes(x = Hour + 0.5))
  
  if (!is_DD) {
    p <- p + annotate("rect", xmin=seq(hr_min, hr_max-12, 24), xmax=seq(hr_min, hr_max-12, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#FFD700", alpha=0.9) + annotate("rect", xmin=seq(hr_min+12, hr_max, 24), xmax=seq(hr_min+12, hr_max, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#191970", alpha=0.9)
  } else {
    p <- p + annotate("rect", xmin=seq(hr_min, hr_max-12, 24), xmax=seq(hr_min, hr_max-12, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#D3D3D3", alpha=0.9) + annotate("rect", xmin=seq(hr_min+12, hr_max, 24), xmax=seq(hr_min+12, hr_max, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#696969", alpha=0.9)
  }
  
  p <- p + geom_hline(yintercept = (0 - ylim2[1]) / (ylim2[2] - ylim2[1]) * (ylim1[2] - ylim1[1]) + ylim1[1], linetype = "dashed", color = "gray50", linewidth = 0.8) + 
    geom_ribbon(aes(ymin = m_v2_scaled - se_v2_scaled, ymax = m_v2_scaled + se_v2_scaled), fill = var2_color, alpha = 0.2) + geom_line(aes(y = m_v2_scaled), color = var2_color, linewidth = 1) + 
    geom_ribbon(aes(ymin = m_sim - se_sim, ymax = m_sim + se_sim), fill = "#2980b9", alpha = 0.3) + geom_line(aes(y = m_sim), color = "#2980b9", linewidth = 1) + 
    scale_x_continuous(breaks = seq(hr_min, hr_max, by=12), limits = c(hr_min, hr_max)) + 
    scale_y_continuous(name = "Strict NREM similarity", limits = ylim1, sec.axis = sec_axis(~ (. - ylim1[1]) / (ylim1[2] - ylim1[1]) * (ylim2[2] - ylim2[1]) + ylim2[1], name = var2_name)) + 
    labs(title = paste0(target_group, " Overlay (", period_name, ")"), x = "Time (hours)") + theme_pub + 
    # 💡 X軸の目盛りだけは黒色に維持する
    theme(axis.title.y.left = element_text(color = "#2980b9"), axis.text.y.left = element_text(color = "#2980b9"), 
          axis.title.y.right = element_text(color = var2_color), axis.text.y.right = element_text(color = var2_color),
          axis.text.x = element_text(color = "black")) +
    force_panelsizes(rows = unit(6, "cm"), cols = unit(15, "cm"))
  
  return(list(plot = p, data = export_over_data))
}

trans_settings <- list(
  list(col = "Trans_Prob_Pws",       name = "Pws (%)",       color = "#AF58BA", prefix = "Pws"),             
  list(col = "Trans_Prob_minus_Psw", name = "-Psw (%)",      color = "#DC267F", prefix = "minus_Psw"),        
  list(col = "Diff_Pws_Psw",         name = "Pws - Psw (%)", color = "#00CD6C", prefix = "Pws_minus_Psw")
)

for (grp in names(datasets)) {
  for (ts in trans_settings) {
    for (pd in periods) {
      res_over <- plot_overlay_timeseries(datasets[[grp]], grp, pd$h_min, pd$h_max, pd$name, ts$col, ts$name, ts$color, pd$is_DD)
      save_plot_to_pdf(res_over$plot, paste0("06_", ts$prefix, "_Overlay_", grp, "_", pd$name, ".pdf"), width = 22, height = 12)
      my_pptx <- add_to_pptx(my_pptx, res_over$plot, paste0(grp, " Overlay: Strict NREM Sim vs ", ts$name, " (", pd$name, ")"))
      
      # ★ プロットデータのCSV保存
      csv_name <- paste0("06_", ts$prefix, "_OverlayData_", grp, "_", pd$name, ".csv")
      write.csv(res_over$data, file.path(output_dir, csv_name), row.names = FALSE, na = "")
    }
  }
}

# =====================================================================
# Step 7: 個体ごとの時系列オーバーレイ作図 (LD & DD)
# =====================================================================
cat("\n[Step 7] Generating Individual Overlay Plots...\n")
plot_individual_overlay <- function(df_ind, df_group, target_mouse, target_group, pd_name, hr_min, hr_max, is_DD, trans_col, trans_name) {
  df_plot <- df_ind %>% filter(Hour >= hr_min & Hour < hr_max)
  if (nrow(df_plot) == 0) return(NULL)
  
  valid_sim <- df_plot$NREM_Sim[is.finite(df_plot$NREM_Sim)]
  if(length(valid_sim) == 0) return(NULL)
  sim_min <- min(valid_sim); sim_max <- max(valid_sim)
  if(sim_min == sim_max) { sim_min <- sim_min - 0.1; sim_max <- sim_max + 0.1 } 
  
  valid_v2 <- df_plot[[trans_col]][is.finite(df_plot[[trans_col]])]
  if(length(valid_v2) == 0) return(NULL)
  min_v2 <- min(valid_v2); max_v2 <- max(valid_v2)
  if(min_v2 == max_v2) { min_v2 <- min_v2 - 1; max_v2 <- max_v2 + 1 } 
  
  margin1 <- (sim_max - sim_min) * 0.15; ylim1 <- c(sim_min - margin1, sim_max + margin1)
  margin2 <- (max_v2 - min_v2) * 0.15; ylim2 <- c(min_v2 - margin2, max_v2 + margin2)
  
  df_plot <- df_plot %>% mutate(Trans_scaled = (!!sym(trans_col) - ylim2[1]) / (ylim2[2] - ylim2[1]) * (ylim1[2] - ylim1[1]) + ylim1[1])
  
  bar_y_max <- ylim1[1] + (ylim1[2] - ylim1[1]) * 0.01
  p <- ggplot(df_plot, aes(x = Hour + 0.5))
  
  if (!is_DD) { 
    p <- p + annotate("rect", xmin=seq(hr_min, hr_max-12, 24), xmax=seq(hr_min, hr_max-12, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#FFD700", alpha=0.9) + annotate("rect", xmin=seq(hr_min+12, hr_max, 24), xmax=seq(hr_min+12, hr_max, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#191970", alpha=0.9)
  } else { 
    p <- p + annotate("rect", xmin=seq(hr_min, hr_max-12, 24), xmax=seq(hr_min, hr_max-12, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#D3D3D3", alpha=0.9) + annotate("rect", xmin=seq(hr_min+12, hr_max, 24), xmax=seq(hr_min+12, hr_max, 24)+12, ymin=-Inf, ymax=bar_y_max, fill="#696969", alpha=0.9) 
  }
  
  p <- p + geom_hline(yintercept = (0 - ylim2[1]) / (ylim2[2] - ylim2[1]) * (ylim1[2] - ylim1[1]) + ylim1[1], linetype = "dashed", color = "gray50", linewidth = 0.8) + 
    geom_line(aes(y = Trans_scaled), color = "#AF58BA", linewidth = 1) + geom_point(aes(y = Trans_scaled), color = "#AF58BA", size = 2) + 
    geom_line(aes(y = NREM_Sim), color = "#0066CC", linewidth = 1) + geom_point(aes(y = NREM_Sim), color = "#0066CC", size = 2) + 
    scale_x_continuous(breaks = seq(hr_min, hr_max, by=12), limits = c(hr_min, hr_max)) + 
    scale_y_continuous(name = "Strict NREM similarity in wake", limits = ylim1, sec.axis = sec_axis(~ (. - ylim1[1]) / (ylim1[2] - ylim1[1]) * (ylim2[2] - ylim2[1]) + ylim2[1], name = trans_name)) + 
    labs(title = paste0(target_group, " [", target_mouse, "] (", pd_name, ")"), x = "Time (hours)") + theme_pub + 
    # 💡 X軸の目盛りだけは黒色に維持する
    theme(axis.title.y.left = element_text(color = "#0066CC"), axis.text.y.left = element_text(color = "#0066CC"), 
          axis.title.y.right = element_text(color = "#AF58BA"), axis.text.y.right = element_text(color = "#AF58BA"),
          axis.text.x = element_text(color = "black")) +
    force_panelsizes(rows = unit(6, "cm"), cols = unit(15, "cm"))
  
  return(p)
}

for (grp in names(datasets)) {
  for (ts in trans_settings) {
    for (m_id in unique(datasets[[grp]]$Mouse_ID)) {
      df_ind <- datasets[[grp]] %>% filter(Mouse_ID == m_id)
      for (pd in periods) {
        p_over <- plot_individual_overlay(df_ind, datasets[[grp]], m_id, grp, pd$name, pd$h_min, pd$h_max, pd$is_DD, ts$col, ts$name)
        if(!is.null(p_over)) ggsave(file.path(ind_out_dir, sprintf("11_%s_IndOverlay_%s_%s_%s.pdf", ts$prefix, grp, m_id, pd$name)), plot = p_over, device = cairo_pdf, width = 22, height = 12, units = "cm", dpi = 300)
      }
    }
  }
}


# =====================================================================
# Step 8: Individual-relative Binning Analysis (Weighted)
# =====================================================================
cat("\n[Step 8] Starting Relative Individual-based Binning Analysis (Detailed Output)...\n")

binned_ind_stats <- list()

for (grp in names(datasets)) {
  df_group_full <- datasets[[grp]] 
  
  for (ts in trans_settings) {
    for (pd in periods) {
      df_group_pd <- df_group_full %>% 
        filter(Hour >= pd$h_min & Hour < pd$h_max) %>%
        filter(is.finite(NREM_Sim) & is.finite(!!sym(ts$col)))
      
      if (nrow(df_group_pd) == 0) next
      
      for (n_bins in c(6, 7, 8)) {
        
        # 💡 個体ごとのランクと使用したエポック数を計算
        df_ind_bin <- df_group_pd %>%
          group_by(Group, Mouse_ID) %>%
          filter(n() > n_bins, max(NREM_Sim, na.rm=TRUE) > min(NREM_Sim, na.rm=TRUE)) %>%
          mutate(Bin_Order = as.integer(cut(NREM_Sim, breaks = n_bins, include.lowest = TRUE))) %>%
          group_by(Group, Mouse_ID, Bin_Order) %>%
          summarise(
            Data_Points = n(), 
            NREM_Sim_Min = min(NREM_Sim, na.rm = TRUE),             # ★ NREM Similarity の範囲(最小値)
            NREM_Sim_Max = max(NREM_Sim, na.rm = TRUE),             # ★ NREM Similarity の範囲(最大値)
            Ind_Mean_Sim = mean(NREM_Sim, na.rm = TRUE), 
            Ind_Trans_Mean = mean(!!sym(ts$col), na.rm = TRUE),     # ★ 遷移確率(Pwsなど)の平均値
            Ind_Trans_SE = sd(!!sym(ts$col), na.rm = TRUE) / sqrt(n()), # ★ 遷移確率の標準誤差
            Total_Wake_Epochs = sum(Wake_Epochs, na.rm = TRUE),
            Total_Sleep_Epochs = sum(Sleep_Epochs, na.rm = TRUE),
            Total_Strict_Wake_Epochs = sum(Strict_Wake_Epochs, na.rm = TRUE),
            .groups = "drop"
          ) %>%
          filter(!is.na(Bin_Order))
        
        if(nrow(df_ind_bin) == 0) next
        
        # グループ全体の平均値の算出 (散布図用)
        df_group_binned <- df_ind_bin %>%
          group_by(Group, Bin_Order) %>%
          summarise(
            N_Mice = n(), 
            Group_Trans_Mean = weighted.mean(Ind_Trans_Mean, w = Data_Points, na.rm = TRUE),
            Group_Trans_SE = sqrt(sum(Data_Points * (Ind_Trans_Mean - Group_Trans_Mean)^2, na.rm = TRUE) / sum(Data_Points, na.rm = TRUE)) / sqrt(n()),
            .groups = "drop"
          ) %>% filter(!is.na(Group_Trans_Mean))
        
        # ★ マウス単位のデータに、グループ全体での該当Rankの平均値・標準誤差も結合してCSV出力
        df_export <- df_ind_bin %>%
          left_join(df_group_binned %>% dplyr::select(Group, Bin_Order, Group_Trans_Mean, Group_Trans_SE), by = c("Group", "Bin_Order"))
        
        # 💡 CSV出力 1: 個体レベルのランク・遷移確率・エポック数・NREM Sim範囲
        write.csv(df_export, file.path(bin_out_dir, sprintf("13_IndRelative_Binned_%s_%s_%s_%dBins_PlotData.csv", ts$prefix, grp, pd$name, n_bins)), row.names = FALSE, na = "")
        
        if (nrow(df_group_binned) >= 3 && sd(df_group_binned$Bin_Order) > 0 && sd(df_group_binned$Group_Trans_Mean) > 0) {
          
          # 💡 詳細な相関検定
          cor_res <- cor.test(df_group_binned$Bin_Order, df_group_binned$Group_Trans_Mean, method = "spearman", exact = FALSE)
          r_val <- cor_res$estimate
          p_val <- cor_res$p.value
          t_stat <- cor_res$statistic
          df_stat <- cor_res$parameter
          
          sig <- if(p_val < 0.001) "***" else if(p_val < 0.01) "**" else if(p_val < 0.05) "*" else "n.s."
          p_label <- format_pval_text(p_val)
          annot_text <- sprintf("rho = %.3f\n%s", r_val, p_label)
          mic_val <- minerva::mine(df_group_binned$Bin_Order, df_group_binned$Group_Trans_Mean)$MIC
          
          pt_color <- case_when(
            grp == "WT" & grepl("LD", pd$name) ~ "#708090",
            grp == "WT" & grepl("DD", pd$name) ~ "#4F4F4F",
            grp == "Bmal1" & grepl("LD", pd$name) ~ "#648FFF",
            grp == "Bmal1" & grepl("DD", pd$name) ~ "#2D4B9A",
            TRUE ~ "black"
          )
          
          p <- ggplot(df_group_binned, aes(x = Bin_Order, y = Group_Trans_Mean)) +
            geom_errorbar(aes(ymin = Group_Trans_Mean - Group_Trans_SE, ymax = Group_Trans_Mean + Group_Trans_SE), width = 0.2, color = "gray50", linewidth = 0.8) +
            geom_point(alpha = 0.9, color = pt_color, size = 4) +
            geom_smooth(method = "lm", color = "black", se = FALSE, linetype = "dashed", linewidth = 0.8) +
            annotate("text", 
                     x = (n_bins / 2) + 0.5, y = -Inf,   # X軸中央付近に配置
                     label = annot_text, 
                     hjust = 0,            # 左揃え
                     vjust = -0.5,         # 下枠から浮かす
                     size = 4) +
            labs(title = paste0(grp, " Relative Binned (", pd$name, ", ", n_bins, " Ranges)"), x = paste0("Relative Strict NREM Sim Rank\n(1 = Lowest, ", n_bins, " = Highest)"), y = paste0("Mean ", ts$name)) +
            scale_x_continuous(breaks = 1:n_bins, labels = 1:n_bins) + theme_pub +
            force_panelsizes(rows = unit(6, "cm"), cols = unit(6, "cm"))
          
          ggsave(file.path(output_dir, sprintf("14_Relative_Binned_Scatter_%s_%s_%s_%dBins.pdf", ts$prefix, grp, pd$name, n_bins)), plot = p, width = 12, height = 12, units = "cm", dpi = 300)
          my_pptx <- add_to_pptx(my_pptx, p, paste0(grp, " Relative Binning (", n_bins, " bins): Strict NREM Sim vs ", ts$name, " (", pd$name, ")"))
          
          # 💡 詳細な統計結果をリストに保存 (Spearman correlation format)
          binned_ind_stats[[length(binned_ind_stats) + 1]] <- data.frame(
            "Stats" = paste("Strict NREM Sim vs", ts$name),
            "Statistical tests" = "Spearman correlation",
            "Period" = pd$name,
            "Bins" = as.character(n_bins),
            "Groups" = grp,
            "Rho value" = r_val,
            "MIC" = mic_val,
            "P values" = p_val,
            "Stars" = sig,
            "Mice numer (n)" = as.character(df_group_binned$N_Mice[1]), # 代表して最初のbinのNを記載
            check.names = FALSE, stringsAsFactors = FALSE
          )
        }
      }
    }
  }
}

# 💡 CSV出力 3: 詳細な統計結果を統合して出力
if (length(binned_ind_stats) > 0) {
  final_stats_df <- bind_rows(binned_ind_stats)
  write.csv(final_stats_df, file.path(bin_out_dir, "15_Relative_Binned_Detailed_Stats_LD_DD.csv"), row.names = FALSE, na = "")
}


# =====================================================================
# PPTX エクスポート実行
# =====================================================================
pptx_file <- file.path(output_dir, "Summary_Strict-NREM-Sim_Pws_Psw_Plots.pptx")
print(my_pptx, target = pptx_file)
cat(sprintf("\n✅ All Summary Plots successfully exported to PowerPoint: %s\n", pptx_file))

cat("\n✅ All Transitions Output Complete (Strict Wake version, Cleaned Scatter & Tanh)! Check the output directory.\n")