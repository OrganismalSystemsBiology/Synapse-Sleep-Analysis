# =====================================================================
# 必要なライブラリの読み込み
# =====================================================================
library(tidyverse)
library(reticulate) 
library(car)        
library(ggh4x)      # ★ グラフのパネルサイズ固定用に追加

Sys.setenv(RETICULATE_PYTHON = "/opt/anaconda3/bin/python3")

# =====================================================================
# 0. ローカルデータの読み込み & 0.5 グループ名のリネーム
# =====================================================================
local_filepath <- "/Users/keita/Documents/SNAP25_Figure/psd_info_list_K40E.pkl" 

if (!file.exists(local_filepath)) {
  stop("指定されたファイルが見つかりません。パスが正しいか確認してください。")
}

cat("ローカルデータ (", local_filepath, ") を読み込み中...\n")
data_list <- reticulate::py_load_object(local_filepath)

if (is.null(data_list) || length(data_list) == 0) {
  stop("【エラー】データの読み込みに失敗し、中身が空 (NULL) になりました。")
} else {
  cat(sprintf("データを正常に読み込みました。 (動物数: %d 匹)\n", length(data_list)))
}

data_list <- map(data_list, function(animal) {
  animal$mouse_group <- str_replace(animal$mouse_group, "SNAP25_WT", "WT")
  animal$mouse_group <- str_replace(animal$mouse_group, "SNAP25_K40E", "K40E")
  return(animal)
})
cat("Group names have been renamed.\n")

# =====================================================================
# 1. 保存先ディレクトリの作成とヘルパー関数
# =====================================================================
output_dir <- "outputs"
dir.create(output_dir, showWarnings = FALSE)

save_df_to_csv <- function(df, filename) {
  filepath <- file.path(output_dir, filename)
  write.csv(df, filepath, row.names = FALSE, na = "") # 💡 na="" を追加
  cat(sprintf("Data saved to %s\n", filepath))
}

# ★ PDF保存時のキャンバスサイズをインチからcmに変更し、見切れないよう余裕を持たせました
save_plot_to_pdf <- function(plot_obj, filename, width = 14, height = 14) {
  filepath <- file.path(output_dir, filename)
  ggsave(filepath, plot = plot_obj, device = cairo_pdf, width = width, height = height, units = "cm", dpi = 300)
  cat(sprintf("Plot saved to %s\n", filepath))
}

# ★ 時系列(6x15cmパネル)用のPDF保存設定
save_plot_to_pdf_tc <- function(plot_obj, filename, width = 24, height = 14) {
  filepath <- file.path(output_dir, filename)
  ggsave(filepath, plot = plot_obj, device = cairo_pdf, width = width, height = height, units = "cm", dpi = 300)
  cat(sprintf("Plot saved to %s\n", filepath))
}

# =====================================================================
# 2. グループ名とカラーの固定設定
# =====================================================================
WT_GROUP  <- "WT"
MT_GROUPS <- "K40E" 
PLOT_ORDER <- c(WT_GROUP, MT_GROUPS)

GROUP_COLORS <- c(
  "WT"   = "#708090",
  "K40E" = "#0066CC"
)

# ご指定のステージカラー
STAGE_COLORS <- c('WAKE' = '#DC267F', 'NREM' = '#648FFF', 'REM' = '#FFB000')

# 定数
EPOCH_SEC <- 8
EPOCHS_PER_HR <- 3600 / EPOCH_SEC
MAX_BINS <- 3 * 24 * EPOCHS_PER_HR # 最初の3日間


# =====================================================================
# 3. 共通テーマ・統計検定関数の定義
# =====================================================================
theme_pub <- theme_test() +
  theme(
    text = element_text(colour = "black", size = 14),
    axis.title = element_text(size = 14),
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(colour = "black", size = 14),
    axis.text.y = element_text(colour = "black", size = 14),
    legend.position = "none",
    legend.key.size = unit(0.9, "lines"),
    legend.text = element_text(size = 12),
    legend.title = element_blank(),
    legend.background = element_rect(fill = "transparent"),
    legend.box.spacing = unit(0.05, "cm"),
    panel.border = element_rect(colour = "black", linewidth = 1.2, fill = NA)
  )

perform_smart_stat_test <- function(g1_data, g2_data) {
  g1_data <- na.omit(g1_data)
  g2_data <- na.omit(g2_data)
  
  if (length(g1_data) < 3 || length(g2_data) < 3) {
    res <- wilcox.test(g1_data, g2_data, exact = FALSE)
    return(list(p_val = res$p.value, test_name = "Mann-Whitney U"))
  }
  
  p_norm1 <- shapiro.test(g1_data)$p.value
  p_norm2 <- shapiro.test(g2_data)$p.value
  
  if (p_norm1 >= 0.05 && p_norm2 >= 0.05) {
    df_var <- data.frame(val = c(g1_data, g2_data), group = rep(c("g1", "g2"), c(length(g1_data), length(g2_data))))
    p_var <- car::leveneTest(val ~ group, data = df_var)$`Pr(>F)`[1]
    
    if (p_var >= 0.05) {
      res <- t.test(g1_data, g2_data, var.equal = TRUE)
      return(list(p_val = res$p.value, test_name = "Student's t-test"))
    } else {
      res <- t.test(g1_data, g2_data, var.equal = FALSE)
      return(list(p_val = res$p.value, test_name = "Welch's t-test"))
    }
  } else {
    res <- wilcox.test(g1_data, g2_data, exact = FALSE)
    return(list(p_val = res$p.value, test_name = "Mann-Whitney U"))
  }
}

get_sig_symbol <- function(p) {
  if (p >= 0.05) return("n.s.")
  if (p < 0.001) return("***")
  if (p < 0.01) return("**")
  return("*")
}

# ★ 新規追加: 指定のフォーマットで全検定結果をまとめる関数
generate_comprehensive_stats_table <- function(x_mut, x_wt, stat_name, mut_name="SD", wt_name="Control") {
  safe_p <- function(expr) {
    res <- tryCatch(expr, error = function(e) NA)
    if(is.list(res) && "p.value" %in% names(res)) return(res$p.value)
    return(NA_real_)
  }
  
  x_mut <- na.omit(as.numeric(x_mut))
  x_wt  <- na.omit(as.numeric(x_wt))
  n_mut <- length(x_mut)
  n_wt  <- length(x_wt)
  
  if (n_mut < 2 || n_wt < 2) return(data.frame()) 
  
  p_shap_mut <- safe_p(shapiro.test(x_mut))
  p_shap_wt  <- safe_p(shapiro.test(x_wt))
  p_f        <- safe_p(var.test(x_mut, x_wt))
  p_wilcox   <- safe_p(wilcox.test(x_mut, x_wt, exact = FALSE))
  p_welch    <- safe_p(t.test(x_mut, x_wt, var.equal = FALSE))
  p_student  <- safe_p(t.test(x_mut, x_wt, var.equal = TRUE))
  
  get_stars <- function(p) {
    if (is.na(p) || is.null(p)) return("")
    if (p < 0.001) return("***")
    if (p < 0.01) return("**")
    if (p < 0.05) return("*")
    return("")
  }
  
  df <- data.frame(
    "Stats" = c(stat_name, rep("", 5)),
    "Statistical tests" = c("Shapiro test", "Shapiro test", "F-test", "Wilcoxon test", "Welch's t-test", "Student's t-test"),
    "Groups" = c(mut_name, wt_name, "", mut_name, mut_name, mut_name),
    " " = c("", "", "", wt_name, wt_name, wt_name),
    "P values" = as.numeric(c(p_shap_mut, p_shap_wt, p_f, p_wilcox, p_welch, p_student)),
    "Stars" = c(get_stars(p_shap_mut), get_stars(p_shap_wt), get_stars(p_f), get_stars(p_wilcox), get_stars(p_welch), get_stars(p_student)),
    "  " = rep("", 6),
    "Groups " = c(mut_name, wt_name, rep("", 4)),
    "Mice numer (n)" = as.character(c(n_mut, n_wt, rep("", 4))),
    check.names = FALSE, stringsAsFactors = FALSE
  )
  return(df)
}

# --- B. Z-score ステージ別スペクトル (自分自身のZ-score; 既存機能) ---
plot_individual_zscore_stages <- function(data) {
  cat("\nCalculating Individual Z-scored spectra for each stage...\n")
  res_list <- list()
  for (animal in data) {
    limit <- min(nrow(animal$raw), MAX_BINS)
    raw_spec <- animal$raw[1:limit, ]
    stages <- toupper(trimws(as.character(unlist(animal$stage_call))[1:limit]))
    freqs <- as.numeric(unlist(animal$freq_bins))
    
    total_pwr <- rowSums(raw_spec, na.rm = TRUE)
    norm_spec <- sweep(raw_spec, 1, total_pwr + 1e-10, "/") * 100
    log_spec <- log10(norm_spec + 1e-10)
    z_spec <- sweep(sweep(log_spec, 2, colMeans(log_spec, na.rm = TRUE), "-"), 2, apply(log_spec, 2, stats::sd, na.rm = TRUE) + 1e-10, "/")
    
    for (st in c('WAKE', 'NREM', 'REM')) {
      mask <- stages == st
      if (sum(mask, na.rm = TRUE) > 0) res_list[[length(res_list)+1]] <- tibble(genotype = animal$mouse_group, stage = st, freq = freqs, val = colMeans(z_spec[mask, , drop=FALSE], na.rm=TRUE))
    }
  }
  
  df_sum <- bind_rows(res_list) %>%
    mutate(genotype = factor(genotype, levels = PLOT_ORDER), stage = factor(stage, levels = c("NREM", "REM", "WAKE"))) %>%
    filter(freq <= 30) %>% group_by(genotype, stage, freq) %>%
    summarise(mean = mean(val, na.rm=TRUE), sem = stats::sd(val, na.rm=TRUE)/sqrt(n()), .groups="drop")
  
  for (g in PLOT_ORDER) {
    sub_df <- df_sum %>% filter(genotype == g)
    if(nrow(sub_df) == 0) next
    
    p <- ggplot(sub_df, aes(x = freq, y = mean, color = stage, fill = stage)) +
      geom_hline(yintercept = 0, color = "gray", linetype = "dashed") +
      geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem), alpha = 0.15, color = NA) + geom_line(linewidth = 0.8) +
      scale_color_manual(values = STAGE_COLORS) + scale_fill_manual(values = STAGE_COLORS) +
      labs(title = paste0("Z-score PSD (", g, ")"), x = "Frequency (Hz)", y = "Z-score PSD") +
      scale_x_continuous(breaks = seq(0, 30, by = 10)) + scale_y_continuous (limits = c(-1.5, 1.5), breaks = seq(-1.5, 1.5, by = 0.5)) +
      theme_pub + theme(legend.position = "right") +
      force_panelsizes(rows = unit(6, "cm"), cols = unit(6, "cm")) 
    
    save_plot_to_pdf(p, paste0("Zscore_Stages_", g, ".pdf"), width = 14, height = 14)
    # ★ グラフ描画データの保存
    save_df_to_csv(sub_df, paste0("Zscore_Stages_", g, "_PlotData.csv"))
  }
}

# --- ★ 箱ひげ図版: NREM Sim 用 ---
plot_Projection_bar_by_phase_with_stats <- function(df, phase = 'allday', title_prefix = 'AllDay') {
  phase_suffix <- switch(phase, "allday" = "24 h", "light" = "Light", "dark" = "Dark", "")
  
  m <- list(
    col = paste0("nrem_sim_", phase), 
    title = "", 
    y_label = "NREM similarity in wake", 
    safe_name = "NREM_Sim",
    y_lim = c(-0.8, 0), 
    y_breaks = round(seq(-0.8, 0, by = 0.2), 1)
  )
  
  df_clean <- df %>% filter(!is.na(!!sym(m$col)))
  if(nrow(df_clean) == 0) return(NULL)
  
  full_title <- paste(m$title, phase_suffix)
  
  p <- ggplot(df_clean, aes(x = genotype, y = !!sym(m$col))) +
    geom_boxplot(aes(fill = genotype), outlier.shape = NA, alpha = 0.4, width = 0.5, color = "black") +
    geom_jitter(aes(color = genotype), width = 0.15, size = 2, alpha = 0.7) +
    scale_fill_manual(values = GROUP_COLORS) + 
    scale_color_manual(values = GROUP_COLORS) +
    labs(title = full_title, x = "", y = m$y_label) + 
    theme_pub + scale_y_continuous(limits = m$y_lim, breaks = m$y_breaks) +
    force_panelsizes(rows = unit(6, "cm"), cols = unit(6, "cm")) 
  
  wt_data <- df_clean %>% filter(genotype == WT_GROUP) %>% pull(m$col)
  
  # ★ 詳細フォーマット用のリスト
  detailed_stats_list <- list()
  
  y_top_limit <- m$y_lim[2]
  y_offset <- (m$y_lim[2] - m$y_lim[1]) * 0.1
  
  for (i in seq_along(MT_GROUPS)) {
    mt <- MT_GROUPS[i]
    mt_data <- df_clean %>% filter(genotype == mt) %>% pull(m$col)
    if (length(wt_data) < 2 || length(mt_data) < 2) next
    
    stat_res <- perform_smart_stat_test(wt_data, mt_data)
    sig_sym <- get_sig_symbol(stat_res$p_val)
    
    # ★ 詳細フォーマットテーブル生成
    stat_table <- generate_comprehensive_stats_table(mt_data, wt_data, stat_name = paste0(m$y_label, " (", phase_suffix, ")"), mut_name = mt, wt_name = WT_GROUP)
    detailed_stats_list[[length(detailed_stats_list) + 1]] <- stat_table
    
    # ★ 警告回避のため annotate を使用
    if (sig_sym != "n.s.") {
      y_bracket <- y_top_limit - (y_offset * 0.5) + (y_offset * 0.8 * (i - 1))
      p <- p + 
        annotate("segment", x = 1, xend = i + 1, y = y_bracket, yend = y_bracket, color = "black", linewidth = 0.6) +
        annotate("segment", x = 1, xend = 1, y = y_bracket, yend = y_bracket - y_offset*0.2, color = "black", linewidth = 0.6) +
        annotate("segment", x = i + 1, xend = i + 1, y = y_bracket, yend = y_bracket - y_offset*0.2, color = "black", linewidth = 0.6) +
        annotate("text", x = (1 + i + 1)/2, y = y_bracket + y_offset*0.1, label = sig_sym, size = 6, fontface = "bold")
    }
  }
  
  save_plot_to_pdf(p, paste0("Box_", m$safe_name, "_", title_prefix, ".pdf"), width = 14, height = 14)
  
  # ★ グラフ描画データのCSV出力
  save_df_to_csv(df_clean, paste0("Box_", m$safe_name, "_", title_prefix, "_PlotData.csv"))
  
  return(if(length(detailed_stats_list) > 0) bind_rows(detailed_stats_list) else NULL)
}

# --- C. NREM Projection & Cosine Sim ---
analyze_nrem_projection_individual <- function(data) {
  cat("\nCalculating Projection Metrics & WT-Ref Z-scores...\n")
  proc_list <- list(); wt_log_list <- list(); wt_stages_list <- list()
  
  for (animal in data) {
    limit <- min(nrow(animal$raw), MAX_BINS)
    raw_spec <- animal$raw[1:limit, ]; stages <- toupper(trimws(as.character(unlist(animal$stage_call))[1:limit]))
    freqs <- as.numeric(unlist(animal$freq_bins))
    log_spec <- log10(sweep(raw_spec, 1, rowSums(raw_spec, na.rm = TRUE) + 1e-10, "/") * 100 + 1e-10)
    
    proc_list[[length(proc_list)+1]] <- list(gid = animal$mouse_group, mid = animal$mouse_id, stages = stages, log_spec = log_spec, freqs = freqs)
    if (animal$mouse_group == WT_GROUP) { wt_log_list[[length(wt_log_list)+1]] <- log_spec; wt_stages_list[[length(wt_stages_list)+1]] <- stages }
  }
  
  freqs_arr <- proc_list[[1]]$freqs
  wt_log_all <- do.call(rbind, wt_log_list)
  wt_stages_all <- do.call(c, wt_stages_list)
  
  global_wt_mean <- colMeans(wt_log_all, na.rm = TRUE)
  global_wt_std <- apply(wt_log_all, 2, stats::sd, na.rm = TRUE)
  global_wt_std <- ifelse(global_wt_std > 0, global_wt_std, 1.0)
  
  wt_z_all <- sweep(sweep(wt_log_all, 2, global_wt_mean, "-"), 2, global_wt_std, "/")
  wt_nrem_z <- wt_z_all[wt_stages_all == 'NREM', , drop = FALSE]
  wt_wake_z <- wt_z_all[wt_stages_all == 'WAKE', , drop = FALSE]
  
  nrem_vec <- colMeans(wt_nrem_z, na.rm = TRUE)
  wake_vec <- colMeans(wt_wake_z, na.rm = TRUE)
  nrem_unit <- nrem_vec / sqrt(sum(nrem_vec^2))
  
  # Projection Calculations
  ref_nrem_proj <- as.numeric(wt_nrem_z %*% nrem_unit)
  ref_wake_proj <- as.numeric(wt_wake_z %*% nrem_unit)
  mean_ref_wake_cos <- mean(ref_wake_proj / (sqrt(rowSums(wt_wake_z^2)) + 1e-10), na.rm=TRUE)
  mean_ref_nrem_cos <- mean(ref_nrem_proj / (sqrt(rowSums(wt_nrem_z^2)) + 1e-10), na.rm=TRUE)
  
  all_scores <- list(); timecourse <- list(); phase_summary <- list()
  wt_ref_z_list <- list(); log10_psd_list <- list() 
  
  for (d in proc_list) {
    limit_idx <- length(d$stages)
    # WT基準の平均・標準偏差を使ってZ-score化
    z_spec <- sweep(sweep(d$log_spec, 2, global_wt_mean, "-"), 2, global_wt_std, "/")
    proj_all <- as.numeric(z_spec %*% nrem_unit)
    nrem_sim_all <- proj_all / (sqrt(rowSums(z_spec^2)) + 1e-10)
    
    # 💡 Strict Wake Mask の作成
    wake_mask <- d$stages == 'WAKE'
    prev_wake <- dplyr::lag(wake_mask, default = TRUE)
    next_wake <- dplyr::lead(wake_mask, default = TRUE)
    strict_wake_mask <- wake_mask & prev_wake & next_wake
    
    if (sum(strict_wake_mask) > 0) {
      all_scores[[length(all_scores)+1]] <- tibble(genotype = d$gid, mouse_id = d$mid, nrem_sim = nrem_sim_all[strict_wake_mask])
    }
    
    # WT基準のZ-scoreスペクトルと、生Log10PSDをステージごとに記録
    for (st in c('WAKE', 'NREM', 'REM')) {
      st_mask <- d$stages == st
      if (sum(st_mask, na.rm = TRUE) > 0) {
        wt_ref_z_list[[length(wt_ref_z_list)+1]] <- tibble(
          genotype = d$gid, stage = st, freq = freqs_arr, val = colMeans(z_spec[st_mask, , drop=FALSE], na.rm=TRUE)
        )
        log10_psd_list[[length(log10_psd_list)+1]] <- tibble(
          genotype = d$gid, stage = st, freq = freqs_arr, val = colMeans(d$log_spec[st_mask, , drop=FALSE], na.rm=TRUE)
        )
      }
    }
    
    # Timecourse extraction (Strict Wake)
    for (h in 0:((length(nrem_sim_all) %/% EPOCHS_PER_HR)-1)) {
      idx_range <- (h * EPOCHS_PER_HR + 1):((h + 1) * EPOCHS_PER_HR)
      h_mask <- strict_wake_mask[idx_range]
      if (sum(h_mask) > 0) {
        timecourse[[length(timecourse)+1]] <- tibble(
          genotype = d$gid, hour = h, 
          m_cos = mean(nrem_sim_all[idx_range][h_mask], na.rm=TRUE)
        )
      }
    }
    
    hours_in_day <- ((seq_len(limit_idx) - 1) %/% EPOCHS_PER_HR) %% 24
    masks <- list(allday = rep(TRUE, limit_idx), light = hours_in_day < 12, dark = hours_in_day >= 12)
    
    res_phase <- tibble(genotype = d$gid, mouse_id = d$mid)
    for (phase in names(masks)) {
      phase_wake_mask <- masks[[phase]] & strict_wake_mask # 💡 Strict Wakeを使用
      if (sum(phase_wake_mask) > 0) {
        res_phase[[paste0("nrem_sim_", phase)]] <- mean(nrem_sim_all[phase_wake_mask], na.rm = TRUE)
      } else {
        res_phase[[paste0("nrem_sim_", phase)]] <- NA
      }
    }
    phase_summary[[length(phase_summary)+1]] <- res_phase
  }
  
  # WT基準Z-scoreのグラフ出力 (グループ別)
  df_wt_ref_z <- bind_rows(wt_ref_z_list) %>%
    mutate(genotype = factor(genotype, levels = PLOT_ORDER), stage = factor(stage, levels = c("NREM", "REM", "WAKE"))) %>%
    filter(freq <= 30) %>% group_by(genotype, stage, freq) %>%
    summarise(mean = mean(val, na.rm=TRUE), sem = stats::sd(val, na.rm=TRUE)/sqrt(n()), .groups="drop")
  
  for (g in PLOT_ORDER) {
    sub_df <- df_wt_ref_z %>% filter(genotype == g)
    if(nrow(sub_df) == 0) next
    
    p_ref_z <- ggplot(sub_df, aes(x = freq, y = mean, color = stage, fill = stage)) +
      geom_hline(yintercept = 0, color = "gray", linetype = "dashed") +
      geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem), alpha = 0.15, color = NA) + geom_line(linewidth = 0.8) +
      scale_color_manual(values = STAGE_COLORS) + scale_fill_manual(values = STAGE_COLORS) +
      labs(title = paste0("WT-Reference Z-score PSD (", g, ")"), x = "Frequency (Hz)", y = "WT-Ref Z-score PSD") +
      scale_x_continuous(breaks = seq(0, 30, by = 10)) + 
      theme_pub + theme(legend.position = "right") +
      force_panelsizes(rows = unit(6, "cm"), cols = unit(6, "cm")) 
    
    save_plot_to_pdf(p_ref_z, paste0("WT_Ref_Zscore_Stages_", g, ".pdf"), width = 14, height = 14)
    # ★ グラフ描画データの保存
    save_df_to_csv(sub_df, paste0("WT_Ref_Zscore_Stages_", g, "_PlotData.csv"))
  }
  
  df_all <- bind_rows(all_scores) %>% mutate(genotype = factor(genotype, levels = PLOT_ORDER))
  df_tc <- bind_rows(timecourse) %>% mutate(genotype = factor(genotype, levels = PLOT_ORDER)) %>%
    group_by(genotype, hour) %>% 
    summarise(mean_cos = mean(m_cos, na.rm=TRUE), se_cos = sd(m_cos, na.rm=TRUE)/sqrt(n()), .groups="drop")
  
  df_proj_phase <- bind_rows(phase_summary) %>% mutate(genotype = factor(genotype, levels = PLOT_ORDER))
  save_df_to_csv(df_proj_phase, "Projection_Phase_Summary.csv")
  save_df_to_csv(df_all, "Projection_AllScores_Raw.csv")
  save_df_to_csv(df_tc, "Projection_Timecourse_Summary.csv")
  
  # ★ 統計検定の詳細フォーマット結果を取得・保存
  res_allday <- plot_Projection_bar_by_phase_with_stats(df_proj_phase, phase = 'allday', title_prefix = '24 h')
  res_light  <- plot_Projection_bar_by_phase_with_stats(df_proj_phase, phase = 'light', title_prefix = 'Light')
  res_dark   <- plot_Projection_bar_by_phase_with_stats(df_proj_phase, phase = 'dark', title_prefix = 'Dark')
  
  all_box_stats <- bind_rows(res_allday, res_light, res_dark)
  if (!is.null(all_box_stats) && nrow(all_box_stats) > 0) {
    save_df_to_csv(all_box_stats, "Stats_Results_Detailed_Summary_Boxplots.csv")
  }
  
  # C-4. Cosine Similarity KDE 
  p_cos_kde <- ggplot(df_all, aes(x=nrem_sim, fill=genotype, color=genotype)) +
    geom_density(alpha=0.2, linewidth=0.7) + geom_vline(xintercept=mean_ref_wake_cos, color="red", linetype="dotted") + geom_vline(xintercept=mean_ref_nrem_cos, color="blue", linetype="dotted") +
    scale_color_manual(values=GROUP_COLORS) + scale_fill_manual(values=GROUP_COLORS) + scale_x_continuous(limits = c(-1, 1), breaks = seq(-1, 1, by = 0.5)) +
    coord_cartesian(xlim=c(-1, 1)) + labs(title="Kernel density estimation", x="NREM similarity in wake", y="Density") + scale_y_continuous(limits = c(0, 4), breaks = seq(0, 4, by = 1)) + theme_pub + theme(legend.position = c(0.82, 0.86)) +
    force_panelsizes(rows = unit(6, "cm"), cols = unit(6, "cm")) 
  
  save_plot_to_pdf(p_cos_kde, "Proj_NREM_Sim_KDE.pdf", width = 14, height = 14)
  # ★ KDEグラフデータの保存
  save_df_to_csv(df_all, "Proj_NREM_Sim_KDE_PlotData.csv")
  
  # C-5. Cosine Similarity Timecourse 
  p_cos_tc <- ggplot(df_tc, aes(x=hour+0.5, y=mean_cos, color=genotype, fill=genotype)) +
    annotate("rect", xmin=seq(0,72,24), xmax=seq(0,72,24)+12, ymin=-0.8, ymax=-0.76, fill="#FFD700", alpha=0.9) +
    annotate("rect", xmin=seq(12,72,24), xmax=seq(12,72,24)+12, ymin=-0.8, ymax=-0.76, fill="#191970", alpha=0.9) +
    geom_hline(yintercept=mean_ref_wake_cos, color="red", linetype="dotted") + geom_ribbon(aes(ymin=mean_cos-se_cos, ymax=mean_cos+se_cos), alpha=0.2, color=NA) +
    geom_line(linewidth=0.8) + geom_point(size = 1.5) +
    scale_color_manual(values=GROUP_COLORS) + scale_fill_manual(values=GROUP_COLORS) +
    scale_x_continuous(limits = c(0, 72), breaks = seq(0, 72, by = 12)) + scale_y_continuous(limits = c(-0.8, 0.2), breaks = seq(-0.8, 0.2, by = 0.2)) +
    labs(title="3-day time series", x="Time (hours)", y="NREM similarity in wake") + theme_pub + theme(legend.position = c(0.85, 0.85)) +
    force_panelsizes(rows = unit(6, "cm"), cols = unit(15, "cm")) 
  
  save_plot_to_pdf_tc(p_cos_tc, "Proj_NREM_Sim_Timecourse.pdf", width = 24, height = 14)
  # ★ 時系列グラフデータの保存
  save_df_to_csv(df_tc, "Proj_NREM_Sim_Timecourse_PlotData.csv")
}


# =====================================================================
# 5. 実行処理
# =====================================================================
cat("\n--- Running all projection analyses ---\n")

# 元のコードで呼び出されていた関数（plot_processing_steps_separated等）が別途定義されている場合はそのまま機能します。
# 呼び出しは維持しておきます。
tryCatch({
  plot_processing_steps_separated(data_list)
}, error = function(e) cat("Note: plot_processing_steps_separated() function not found. Skipping.\n"))

plot_individual_zscore_stages(data_list)
analyze_nrem_projection_individual(data_list)

cat("\n✅ All projection analyses and 1-by-1 PDF plotting completed successfully!\n")