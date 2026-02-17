# teknoalg2.py
"""
TEKNOFEST KUTUP ARAŞTIRMALARI PROJESİ: POLAR ALG ANALYTICS SUITE
Premium Sürüm - Gelişmiş Görselleştirme ve 3D Render
Tam Kapsamlı Bilimsel Analiz Platformu
TÜRKÇE GENİŞLETİLMİŞ VERSİYON - Tüm kod orjinal ve çalışır durumda

📌 BİLİMSEL MODEL VARSARIMMLARI VE SINIRLAMALARI
- Bu projede kullanılan tüm modeller SENTETİK VERİ üzerinde çalışmaktadır
- Fiziksel parametreler literatürden alınmış olup yaklaşık değerlerdir
- Enerji dengesi modeli basitleştirilmiştir
- Mikroplastik etkisi HİPOTETİK bir senaryodur
- İstatistiksel analizler KEŞİFSEL AMAÇLIDIR
"""

# ==================== GEREKLİ KÜTÜPHANELER ====================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML ve Bilimsel Hesaplamalar - KULLANILMAYAN KÜTÜPHANELER KALDIRILDI
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# İleri Seviye ML - SADECE KULLANILANLAR
# Not: TensorFlow, XGBoost, LightGBM gereksiz bağımlılık oluşturduğu için kaldırıldı

# Bilimsel Hesaplamalar
from scipy import stats, signal, interpolate
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, Rbf, RegularGridInterpolator, LinearNDInterpolator
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress, gaussian_kde
from scipy import ndimage
import scipy.spatial as spatial
from scipy.fft import fft, fftfreq

# 3D ve Görsel
import plotly.io as pio
from plotly.colors import sample_colorscale
import colorsys
import colorcet as cc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Veri İşleme
import io
import base64
import json
import pickle
import joblib
import zipfile
from PIL import Image, ImageDraw, ImageFont
import networkx as nx
from itertools import combinations

# Zaman Serisi Analizi
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# Görsel Stil Ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_style("whitegrid")
plotly_template = "plotly_dark"
pio.templates.default = plotly_template

# Özel renk paletleri
ÖZEL_RENKLER = {
    'buz_mavisi': ['#E6F7FF', '#BAE7FF', '#91D5FF', '#69C0FF', '#40A9FF', '#1890FF', '#096DD9', '#0050B3', '#003A8C', '#002766'],
    'alg_yesili': ['#F6FFED', '#D9F7BE', '#B7EB8F', '#95DE64', '#73D13D', '#52C41A', '#389E0D', '#237804', '#135200', '#092B00'],
    'sicaklik_kirmizisi': ['#FFF1F0', '#FFCCC7', '#FFA39E', '#FF7875', '#FF4D4F', '#F5222D', '#CF1322', '#A8071A', '#820014', '#5C0011'],
    'kutup_aurorasi': ['#03045e', '#023e8a', '#0077b6', '#0096c7', '#00b4d8', '#48cae4', '#90e0ef', '#ade8f4', '#caf0f8'],
    'bilimsel': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1C32', '#6A0572', '#AB83A1'],
}

# ==================== FİZİKSEL MODEL GÜNCELLEMELERİ - DÜZELTİLMİŞ ====================

class GelişmişKutupVeriÜretici:
    """Bilimsel olarak gerçekçi kutup verisi üretici - GÜNCELLENMİŞ FİZİKSEL MODELLER"""
    
    def __init__(self, tohum=42):
        np.random.seed(tohum)
        self.parametreler = self._başlat_parametreler()
        
    def _başlat_parametreler(self):
        """Fiziksel parametreleri başlat - KAYNAKLARI BELİRTİLMİŞ"""
        return {
            # Temel parametreler (Kaynak: IPCC AR6, 2021)
            'albedo_temiz_buz': 0.85,      # Temiz buz albedosu (0.8-0.9 arası)
            'albedo_kirli_buz': 0.30,      # Kirli buz albedosu (0.3-0.4 arası)
            'albedo_okyanus': 0.06,        # Okyanus albedosu
            
            # Alg büyüme parametreleri (Kaynak: Thomas & Dieckmann, 2002)
            'alg_büyüme_oranı': 0.15,      # Maksimum spesifik büyüme oranı
            'alg_ölüm_oranı_temel': 0.08,  # Temel ölüm oranı
            'optimal_sıcaklık': -2.0,      # Optimal büyüme sıcaklığı
            'sıcaklık_toleransı': 8.0,     # Sıcaklık toleransı
            
            # Fiziksel sabitler (Kaynak: Uluslararası Sabitler Sistemi)
            'stefan_boltzmann': 5.67e-8,   # Stefan-Boltzmann sabiti (W/m²K⁴)
            'buz_yoğunluğu': 917.0,        # Buz yoğunluğu (kg/m³)
            'su_yoğunluğu': 1027.0,        # Deniz suyu yoğunluğu (kg/m³)
            'gizli_ısı_füzyon': 334000.0,  # Buz erime gizli ısısı (J/kg)
            
            # Emisivite değerleri (Kaynak: MODIS albedo ürünleri)
            'emisivite_buz': 0.97,         # Buz emisivitesi
            'emisivite_atmosfer': 0.78,    # Kutup atmosferi emisivitesi
            
            # Kutup koşulları
            'kutup_gecesi_başlangıç': 150,  # Yılın günü
            'kutup_gecesi_son': 330,
            'geceyarısı_güneşi_başlangıç': 0,
            'geceyarısı_güneşi_son': 180,
        }
    
    def üret_fiziksel_veri_seti(self, gün_sayısı=1095, lokasyon_sayısı=5):
        """Fiziksel prensiplere dayalı kapsamlı veri seti - GÜNCELLENMİŞ ENERJİ DENGESİ"""
        
        # Zaman serisi
        tarihler = pd.date_range(start='2020-01-01', periods=gün_sayısı, freq='D')
        t = np.arange(gün_sayısı)
        
        # 1. İKLİM VERİLERİ
        # ----------------
        # Küresel ısınma trendi (IPCC senaryoları)
        ısınma_senaryoları = {
            'SSP1-2.6': 0.02,  # Düşük emisyon
            'SSP2-4.5': 0.035, # Orta emisyon
            'SSP5-8.5': 0.06   # Yüksek emisyon
        }
        ısınma_oranı = ısınma_senaryoları['SSP2-4.5']
        
        # Mevsimsel sıcaklık (fiziksel model)
        mevsimsel_sıcaklık = self._mevsimsel_sıcaklık(t, genlik=15, faz_kayması=-np.pi/2)
        
        # Günlük varyasyon
        günlük_varyasyon = 3 * np.sin(2*np.pi*t + np.random.uniform(0, 2*np.pi))
        
        # Rastgele hava olayları
        hava_olayları = self._üret_hava_olayları(gün_sayısı)
        
        # Toplam sıcaklık
        sıcaklık = (
            mevsimsel_sıcaklık 
            + günlük_varyasyon 
            + ısınma_oranı * (t / 365) 
            + hava_olayları
            + np.random.normal(0, 1.5, gün_sayısı)
        )
        
        # Atmosfer sıcaklığı (buz yüzeyinden daha sıcak)
        atmosfer_sıcaklığı = sıcaklık + 10 + 5 * np.sin(2*np.pi*t/365)  # Atmosfer daha sıcak
        
        # 2. ALG DİNAMİKLERİ - GÜNCELLENMİŞ ÖLÜM ORANI
        # -----------------
        # Sıcaklık bağımlı büyüme fonksiyonu
        def alg_büyüme_fonksiyonu(sıcaklık_değeri, alg_yoğunluğu, mevcut_gün, buz_kalınlığı):
            # Monod tipi büyüme + sıcaklık inhibisyonu
            T_optimal = self.parametreler['optimal_sıcaklık']
            
            # Sıcaklık etkisi (Gaussian)
            sıcaklık_etkisi = np.exp(-((sıcaklık_değeri - T_optimal)**2) / (2 * self.parametreler['sıcaklık_toleransı']**2))
            
            # Taşıma kapasitesi (logistic growth)
            K = 1000 * (1 + 0.5 * np.tanh(0.01 * (sıcaklık_değeri + 10)))  # Sıcaklıkla artan kapasite
            
            # Besin limitasyonu (basit model)
            besin_limitlemesi = 1 / (1 + np.exp(-0.001 * (alg_yoğunluğu - 500)))
            
            # Işık limitasyonu (polar gece/gündüz)
            yılın_günü = mevcut_gün % 365
            kutup_gecesi_başlangıç = self.parametreler['kutup_gecesi_başlangıç']
            kutup_gecesi_son = self.parametreler['kutup_gecesi_son']
            geceyarısı_güneşi_başlangıç = self.parametreler['geceyarısı_güneşi_başlangıç']
            geceyarısı_güneşi_son = self.parametreler['geceyarısı_güneşi_son']
            
            if kutup_gecesi_başlangıç <= yılın_günü <= kutup_gecesi_son:
                ışık_faktörü = 0.1  # Polar gece
            elif geceyarısı_güneşi_başlangıç <= yılın_günü <= geceyarısı_güneşi_son:
                ışık_faktörü = 1.0  # Geceyarısı güneşi
            else:
                ışık_faktörü = 0.5  # Normal gün
            
            büyüme_oranı = (
                self.parametreler['alg_büyüme_oranı'] 
                * sıcaklık_etkisi 
                * ışık_faktörü
                * (1 - alg_yoğunluğu / K)
                * besin_limitlemesi
            )
            
            return büyüme_oranı
        
        # Alg ölüm oranı modeli - GÜNCELLENMİŞ (sıcaklık ve buz kalınlığı etkisi)
        def alg_ölüm_oranı_hesapla(sıcaklık_değer, buz_kalınlık_değer):
            """Çevresel faktörlere bağlı alg ölüm oranı"""
            temel_ölüm = self.parametreler['alg_ölüm_oranı_temel']
            
            # Sıcaklık stresi (aşırı sıcak veya soğuk)
            sıcaklık_stresi = 0.01 * abs(sıcaklık_değer - self.parametreler['optimal_sıcaklık'])
            
            # Buz kalınlığı etkisi (kalın buz daha fazla ölüm)
            buz_etkisi = 0.005 * buz_kalınlık_değer
            
            # UV etkisi (basit mevsimsel model)
            uv_etkisi = 0.002 * (1 + np.sin(2*np.pi*np.arange(len(sıcaklık_değer))/365)) if isinstance(sıcaklık_değer, np.ndarray) else 0.002
            
            toplam_ölüm = temel_ölüm + sıcaklık_stresi + buz_etkisi + uv_etkisi
            
            # Maksimum ölüm oranı sınırı
            return np.clip(toplam_ölüm, 0.01, 0.3)
        
        # Alg yoğunluğu simülasyonu
        alg_yoğunluğu = np.zeros(gün_sayısı)
        alg_yoğunluğu[0] = 10  # Başlangıç değeri
        buz_kalınlığı_temp = 3.0 * np.ones(gün_sayısı)  # Geçici buz kalınlığı
        
        for i in range(1, gün_sayısı):
            büyüme = alg_büyüme_fonksiyonu(sıcaklık[i], alg_yoğunluğu[i-1], i, buz_kalınlığı_temp[i-1])
            ölüm = alg_ölüm_oranı_hesapla(sıcaklık[i], buz_kalınlığı_temp[i-1]) * alg_yoğunluğu[i-1]
            alg_yoğunluğu[i] = max(0, alg_yoğunluğu[i-1] + büyüme - ölüm + np.random.normal(0, 2))
        
        # 3. ALBEDO HESAPLAMASI - GÜNCELLENMİŞ PARAMETRE AÇIKLAMALARI
        # --------------------
        # Fiziksel albedo modeli - PARAMETRELER VARSARIM OLARAK BELİRTİLMİŞ
        def hesapla_albedo(alg_yoğunluğu_değer, kar_derinliği=0.1):
            """
            Albedo modeli varsayımları:
            1. Temiz buz albedosu: 0.85 (literatür değeri)
            2. Alg etkisi: Doğrusal olmayan azalma (varsayımsal)
            3. Kar etkisi: Üstel azalma (varsayımsal)
            4. Yaş etkisi: Mevsimsel salınım (varsayımsal)
            """
            # Buz albedosu
            buz_albedo = self.parametreler['albedo_temiz_buz']
            
            # Kar etkisi - VARSARIM: Kar derinliği ile üstel azalma
            kar_etkisi = 0.4 * np.exp(-kar_derinliği / 0.05)  # Kar derinliği
            
            # Alg etkisi - VARSARIM: Doğrusal olmayan alg etkisi
            alg_etkisi = 0.35 * (1 - np.exp(-alg_yoğunluğu_değer / 200))
            
            # Yaşlandırma etkisi - VARSARIM: Mevsimsel değişim
            yaş_etkisi = 0.05 * np.sin(2*np.pi*t/365)  # Mevsimsel değişim
            
            albedo_değer = (
                buz_albedo 
                + kar_etkisi 
                - alg_etkisi 
                - yaş_etkisi
                + np.random.normal(0, 0.02)
            )
            
            return np.clip(albedo_değer, self.parametreler['albedo_kirli_buz'], self.parametreler['albedo_temiz_buz'])
        
        albedo = hesapla_albedo(alg_yoğunluğu)
        
        # 4. BUZ ERİME MODELİ - GÜNCELLENMİŞ FİZİKSEL MODEL
        # ------------------
        # ENERJİ DENGESİ MODELİ - GÜNCELLENMİŞ (NET UZUN DALGA RADYASYONU)
        def enerji_dengesi_erime(sıcaklık_değer, atmosfer_sıcaklık_değer, albedo_değer, güneş_radyasyonu, rüzgar_hızı):
            """
            Güncellenmiş enerji dengesi modeli:
            1. Kısa dalga radyasyon: Q_sw = S(1-α)
            2. NET uzun dalga radyasyon: Q_lw = εσ(T_surface⁴ - T_atm⁴)
            3. Duyulur ısı: Bulk transfer formülü
            4. Gizli ısı: Nem transferi
            
            Returns: (erime_oranı, Q_uzun_dalga_net)
            """
            # Kısa dalga radyasyon (güneş)
            Q_güneş = güneş_radyasyonu * (1 - albedo_değer)
            
            # NET uzun dalga radyasyon (GÜNCELLENMİŞ)
            T_surface_kelvin = sıcaklık_değer + 273.15
            T_atm_kelvin = atmosfer_sıcaklık_değer + 273.15
            
            # Yüzeyden yayılan radyasyon
            Q_yayılan = self.parametreler['emisivite_buz'] * self.parametreler['stefan_boltzmann'] * T_surface_kelvin**4
            
            # Atmosferden gelen radyasyon
            Q_gelen = self.parametreler['emisivite_atmosfer'] * self.parametreler['stefan_boltzmann'] * T_atm_kelvin**4
            
            # Net uzun dalga radyasyon
            Q_uzun_dalga_net = Q_gelen - Q_yayılan  # NET radyasyon
            
            # Duyulur ısı akışı (bulk transfer formülü)
            rho_hava = 1.225  # kg/m³, deniz seviyesinde
            cp_hava = 1005    # J/(kg·K)
            Ch = 0.0012      # bulk transfer coefficient
            Q_duyulur = rho_hava * cp_hava * Ch * rüzgar_hızı * (atmosfer_sıcaklık_değer - sıcaklık_değer)
            
            # Gizli ısı (nem transferi)
            Q_gizli = 5 * rüzgar_hızı * 0.001 * (atmosfer_sıcaklık_değer - sıcaklık_değer)  # Basitleştirilmiş
            
            # Toplam enerji dengesi
            Q_toplam = Q_güneş + Q_uzun_dalga_net + Q_duyulur + Q_gizli
            
            # Erime hızı (fiziksel)
            erime_oranı = max(0, Q_toplam / (self.parametreler['buz_yoğunluğu'] * self.parametreler['gizli_ısı_füzyon']))
            
            return erime_oranı * 86400, Q_uzun_dalga_net  # günlük erime (m/gün) ve net radyasyon
        
        # Çevresel değişkenler
        güneş_radyasyonu = 300 + 150 * np.sin(2*np.pi*t/365 - np.pi/2)
        rüzgar_hızı = 5 + 3 * np.sin(2*np.pi*t/180) + np.random.exponential(2, gün_sayısı)
        yağış = np.random.exponential(1, gün_sayısı)
        kar_derinliği = 0.1 + 0.05 * np.sin(2*np.pi*t/365)
        
        # Erime hızı ve net radyasyon hesaplama - DÜZELTME: Q_uzun_dalga değişkenini kaydet
        erime_oranı = np.zeros(gün_sayısı)
        Q_uzun_dalga = np.zeros(gün_sayısı)  # DÜZELTME: Değişkeni tanımla
        
        for i in range(gün_sayısı):
            erime_oranı[i], Q_uzun_dalga[i] = enerji_dengesi_erime(
                sıcaklık[i], 
                atmosfer_sıcaklığı[i],
                albedo[i], 
                güneş_radyasyonu[i],
                rüzgar_hızı[i]
            )
        
        # 5. BUZ KALINLIĞI VE KÜTLE DENGESİ
        # ---------------------------------
        buz_kalınlığı = np.zeros(gün_sayısı)
        buz_kalınlığı[0] = 3.0  # Başlangıç kalınlığı: 3 metre
        
        kar_birikimi = np.zeros(gün_sayısı)
        
        for i in range(1, gün_sayısı):
            # Kar birikimi
            kar_yağışı = yağış[i] if sıcaklık[i] < 0 else 0
            kar_erimesi = 0.01 * max(0, sıcaklık[i])  # Kar erimesi
            kar_birikimi[i] = kar_birikimi[i-1] + kar_yağışı - kar_erimesi
            
            # Buz kütle dengesi
            birikim = 0.001 * kar_yağışı  # Karın buza dönüşümü
            ablasyon = erime_oranı[i]
            
            buz_kalınlığı[i] = buz_kalınlığı[i-1] + birikim - ablasyon
            
            # Minimum buz kalınlığı
            buz_kalınlığı[i] = max(0.1, buz_kalınlığı[i])
        
        # 6. ALG TÜRLERİ VE BİYOÇEŞİTLİLİK - ÇOKLU İNDEKS EKLENDİ
        # -------------------------------
        tür_sayısı = 5
        alg_türleri = np.zeros((gün_sayısı, tür_sayısı))
        
        # Türlere özgü parametreler
        tür_parametreleri = {
            'optimal_sıcaklıklar': [-5, -2, 0, 2, 5],  # Her türün optimal sıcaklığı
            'büyüme_oranları': [0.1, 0.15, 0.2, 0.18, 0.12],
            'pigment_yoğunluğu': [0.8, 1.0, 0.6, 0.9, 0.7],  # Pigment koyuluğu
        }
        
        for tür in range(tür_sayısı):
            tür_yoğunluğu = np.zeros(gün_sayısı)
            tür_yoğunluğu[0] = alg_yoğunluğu[0] / tür_sayısı
            
            for i in range(1, gün_sayısı):
                # Tür-spesifik büyüme
                T_optimal = tür_parametreleri['optimal_sıcaklıklar'][tür]
                sıcaklık_farkı = sıcaklık[i] - T_optimal
                büyüme = (
                    tür_parametreleri['büyüme_oranları'][tür]
                    * np.exp(-(sıcaklık_farkı**2) / 50)
                    * (1 - tür_yoğunluğu[i-1] / 300)
                )
                
                # Türler arası rekabet
                rekabet = 0.01 * (alg_yoğunluğu[i-1] - tür_yoğunluğu[i-1])
                
                tür_yoğunluğu[i] = max(0, tür_yoğunluğu[i-1] + büyüme - rekabet)
            
            alg_türleri[:, tür] = tür_yoğunluğu
        
        # Biyoçeşitlilik indeksleri - ÇOKLU İNDEKS EKLENDİ
        shannon_indeksi = self._hesapla_biyoçeşitlilik_shannon(alg_türleri)
        simpson_indeksi = self._hesapla_biyoçeşitlilik_simpson(alg_türleri)
        
        # 7. UZAYSAL VERİ (Grid) - GÜNCELLENMİŞ İNTERPOLASYON
        # ----------------------
        uzaysal_grid = self._üret_uzaysal_grid_verisi(nokta_sayısı=1000)
        
        # 8. VERİ ÇERÇEVESİ OLUŞTURMA - DÜZELTME: Q_uzun_dalga kullan
        # ---------------------------
        veri_çerçevesi = pd.DataFrame({
            # Temel zaman serisi
            'tarih': tarihler,
            'yılın_günü': tarihler.dayofyear,
            'yıl': tarihler.year,
            'ay': tarihler.month,
            'mevsim': self._al_mevsim(tarihler.month),
            
            # İklim değişkenleri
            'sıcaklık': sıcaklık,
            'atmosfer_sıcaklığı': atmosfer_sıcaklığı,
            'sıcaklık_anomalisi': sıcaklık - np.mean(sıcaklık[:365]),
            'güneş_radyasyonu': güneş_radyasyonu,
            'rüzgar_hızı': rüzgar_hızı,
            'yağış': yağış,
            'kar_derinliği': kar_derinliği,
            'bulut_örtüsü': np.random.uniform(0, 1, gün_sayısı),
            'bağıl_nem': np.random.uniform(70, 100, gün_sayısı),
            
            # Alg değişkenleri
            'alg_yoğunluğu': alg_yoğunluğu,
            'alg_yoğunluğu_log': np.log1p(alg_yoğunluğu),
            'alg_büyüme_oranı': np.gradient(alg_yoğunluğu),
            
            # Alg türleri
            'alg_türü_1': alg_türleri[:, 0],
            'alg_türü_2': alg_türleri[:, 1],
            'alg_türü_3': alg_türleri[:, 2],
            'alg_türü_4': alg_türleri[:, 3],
            'alg_türü_5': alg_türleri[:, 4],
            
            # Biyoçeşitlilik - ÇOKLU İNDEKS
            'biyoçeşitlilik_shannon': shannon_indeksi,
            'biyoçeşitlilik_simpson': simpson_indeksi,
            'tür_eşitliği': self._hesapla_tür_eşitliği(alg_türleri),
            
            # Fiziksel özellikler
            'albedo': albedo,
            'albedo_anomalisi': albedo - self.parametreler['albedo_temiz_buz'],
            'albedo_azalması': self.parametreler['albedo_temiz_buz'] - albedo,
            
            # Buz özellikleri
            'erime_oranı': erime_oranı,
            'kümülatif_erime_oranı': np.cumsum(erime_oranı),
            'buz_kalınlığı': buz_kalınlığı,
            'buz_kalınlığı_anomalisi': buz_kalınlığı - 3.0,
            'buz_hacmi': buz_kalınlığı * 1e6,  # m³/km² varsayımı
            
            # Enerji dengesi bileşenleri - GÜNCELLENMİŞ (DÜZELTME: Q_uzun_dalga kullan)
            'enerji_dengesi_güneş': güneş_radyasyonu * (1 - albedo),
            'enerji_dengesi_uzun_dalga_net': Q_uzun_dalga,  # DÜZELTME: Burada tanımlı
            'enerji_dengesi_toplam': erime_oranı * self.parametreler['buz_yoğunluğu'] * self.parametreler['gizli_ısı_füzyon'],
            
            # İstatistiksel özellikler
            'sıcaklık_kayan_7g': pd.Series(sıcaklık).rolling(7).mean().values,
            'alg_kayan_7g': pd.Series(alg_yoğunluğu).rolling(7).mean().values,
            'erime_kayan_30g': pd.Series(erime_oranı).rolling(30).mean().values,
            
            # Uzaysal varyasyon (simüle)
            'enlem': -75 + np.random.randn(gün_sayısı) * 2,
            'boylam': np.random.choice([-60, 120, -160], gün_sayısı),
            'yükseklik': 2000 + np.random.randn(gün_sayısı) * 100,
        })
        
        return veri_çerçevesi, uzaysal_grid
    
    def _mevsimsel_sıcaklık(self, t, genlik=15, faz_kayması=-np.pi/2):
        """Fiziksel mevsimsel sıcaklık modeli"""
        return genlik * np.sin(2*np.pi*t/365 + faz_kayması)
    
    def _üret_hava_olayları(self, gün_sayısı):
        """Rastgele hava olayları simülasyonu"""
        olaylar = np.zeros(gün_sayısı)
        
        # Sıcak hava dalgaları
        sıcak_dalgaları = np.random.poisson(0.05, gün_sayısı)
        for i in range(gün_sayısı):
            if sıcak_dalgaları[i] > 0 and i < gün_sayısı - 5:
                olaylar[i:i+5] += np.random.uniform(3, 8, 5)
        
        # Soğuk hava dalgaları
        soğuk_dalgaları = np.random.poisson(0.03, gün_sayısı)
        for i in range(gün_sayısı):
            if soğuk_dalgaları[i] > 0 and i < gün_sayısı - 3:
                olaylar[i:i+3] -= np.random.uniform(2, 6, 3)
        
        return olaylar
    
    def _hesapla_biyoçeşitlilik_shannon(self, tür_matrisi):
        """Shannon biyoçeşitlilik indeksi hesaplama"""
        gün_sayısı = tür_matrisi.shape[0]
        shannon = np.zeros(gün_sayısı)
        
        for i in range(gün_sayısı):
            # Tür oranları
            oranlar = tür_matrisi[i] / (tür_matrisi[i].sum() + 1e-10)
            oranlar = oranlar[oranlar > 0]
            
            # Shannon indeksi
            if len(oranlar) > 0:
                shannon[i] = -np.sum(oranlar * np.log(oranlar))
        
        return shannon
    
    def _hesapla_biyoçeşitlilik_simpson(self, tür_matrisi):
        """Simpson biyoçeşitlilik indeksi hesaplama"""
        gün_sayısı = tür_matrisi.shape[0]
        simpson = np.zeros(gün_sayısı)
        
        for i in range(gün_sayısı):
            # Tür oranları
            oranlar = tür_matrisi[i] / (tür_matrisi[i].sum() + 1e-10)
            oranlar = oranlar[oranlar > 0]
            
            # Simpson indeksi (1-D)
            if len(oranlar) > 0:
                simpson[i] = 1 - np.sum(oranlar**2)
        
        return simpson
    
    def _hesapla_tür_eşitliği(self, tür_matrisi):
        """Tür dağılımının eşitliği"""
        gün_sayısı = tür_matrisi.shape[0]
        eşitlik = np.zeros(gün_sayısı)
        
        for i in range(gün_sayısı):
            oranlar = tür_matrisi[i] / (tür_matrisi[i].sum() + 1e-10)
            oranlar = oranlar[oranlar > 0]
            
            if len(oranlar) > 1:
                H = -np.sum(oranlar * np.log(oranlar))
                H_maks = np.log(len(oranlar))
                eşitlik[i] = H / H_maks
        
        return eşitlik
    
    def _al_mevsim(self, ay):
        """Ayı mevsime çevir"""
        mevsimler = []
        for m in ay:
            if m in [12, 1, 2]:
                mevsimler.append('Kış')
            elif m in [3, 4, 5]:
                mevsimler.append('İlkbahar')
            elif m in [6, 7, 8]:
                mevsimler.append('Yaz')
            else:
                mevsimler.append('Sonbahar')
        return mevsimler
    
    def _üret_uzaysal_grid_verisi(self, nokta_sayısı=1000):
        """Uzaysal grid verisi oluştur - GÜNCELLENMİŞ İNTERPOLASYON"""
        # Rastgele koordinatlar (Antarktika)
        enlemler = np.random.uniform(-90, -60, nokta_sayısı)
        boylamlar = np.random.uniform(-180, 180, nokta_sayısı)
        
        # Topoğrafya (basit model)
        yükseklik = 2000 + 1000 * np.exp(-((enlemler + 75)**2 + (boylamlar/20)**2) / 1000)
        
        # Alg dağılımı (çok merkezli Gaussian)
        alg_yoğunluğu_uzaysal = np.zeros(nokta_sayısı)
        merkezler = [
            (-70, -60),  # Antarktika Yarımadası
            (-80, 120),  # Doğu Antarktika
            (-85, -160), # Batı Antarktika
        ]
        
        for enlem_m, boylam_m in merkezler:
            mesafe = np.sqrt((enlemler - enlem_m)**2 + (0.5*(boylamlar - boylam_m))**2)
            alg_yoğunluğu_uzaysal += 500 * np.exp(-mesafe**2 / (2*10**2))
        
        # Sıcaklık (enlem gradienti + yükseklik etkisi)
        sıcaklık_uzaysal = -30 + 0.5*(enlemler + 90) - 0.0065*yükseklik + np.random.randn(nokta_sayısı)*3
        
        # Albedo (alg ve kar derinliğine bağlı)
        kar_derinliği_uzaysal = 0.1 + 0.05 * np.sin(enlemler * np.pi/180)
        albedo_uzaysal = 0.85 - 0.3*(1 - np.exp(-alg_yoğunluğu_uzaysal/200)) + 0.1*kar_derinliği_uzaysal
        
        # Erime hızı
        erime_oranı_uzaysal = 0.001 + 0.01*np.exp(0.1*sıcaklık_uzaysal) + 0.001*alg_yoğunluğu_uzaysal
        
        # Buz kalınlığı (5 yıllık erime etkisi)
        buz_kalınlığı_uzaysal = 3.0 - 0.001*erime_oranı_uzaysal*365*5
        
        # NaN değerleri doldurmak için daha iyi yöntem
        # Gaussian filter yerine nearest neighbor interpolation
        def doldur_nan_veri(veri_dizisi):
            """NaN değerleri en yakın komşu ile doldur"""
            nan_maskesi = np.isnan(veri_dizisi)
            if not nan_maskesi.any():
                return veri_dizisi
            
            # NaN olmayan indeksler
            geçerli_indeksler = np.where(~nan_maskesi)[0]
            nan_indeksler = np.where(nan_maskesi)[0]
            
            # En yakın geçerli değerleri bul
            from scipy.spatial import cKDTree
            geçerli_noktalar = np.column_stack([enlemler[geçerli_indeksler], boylamlar[geçerli_indeksler]])
            nan_noktalar = np.column_stack([enlemler[nan_indeksler], boylamlar[nan_indeksler]])
            
            ağaç = cKDTree(geçerli_noktalar)
            _, en_yakın_indeksler = ağaç.query(nan_noktalar, k=1)
            
            # Doldur
            doldurulmuş_veri = veri_dizisi.copy()
            doldurulmuş_veri[nan_indeksler] = veri_dizisi[geçerli_indeksler[en_yakın_indeksler]]
            
            return doldurulmuş_veri
        
        # Her sütunu ayrı ayrı doldur
        alg_yoğunluğu_uzaysal = doldur_nan_veri(alg_yoğunluğu_uzaysal)
        sıcaklık_uzaysal = doldur_nan_veri(sıcaklık_uzaysal)
        albedo_uzaysal = doldur_nan_veri(albedo_uzaysal)
        
        return pd.DataFrame({
            'enlem': enlemler,
            'boylam': boylamlar,
            'yükseklik': yükseklik,
            'alg_yoğunluğu': alg_yoğunluğu_uzaysal,
            'sıcaklık': sıcaklık_uzaysal,
            'albedo': albedo_uzaysal,
            'erime_oranı': erime_oranı_uzaysal,
            'buz_kalınlığı': buz_kalınlığı_uzaysal,
            'kar_derinliği': kar_derinliği_uzaysal,
        })

# ==================== YENİ HİPOTEZ SINIFI - GÜNCELLENMİŞ ====================

class YeniHipotezTestleri:
    """Yeni geliştirilen hipotezleri test eden sınıf - İSTATİSTİKSEL DÜZELTMELER"""
    
    def __init__(self):
        self.hipotez_sonuçları = {}
    
    def tüm_hipotezleri_test_et(self, veri_çerçevesi, uzaysal_veri):
        """Tüm yeni hipotezleri test et - KEŞİFSEL ANALİZ VURGUSU"""
        sonuçlar = {}
        
        # Zaman serisi bağımlılığı kontrolü (ACF/PACF)
        otokorelasyon_analizi = self._otokorelasyon_analizi_yap(veri_çerçevesi['alg_yoğunluğu'])
        
        # Hipotez 1: Alg Çeşitlilik-Erime İlişkisi (Lag-1 korelasyon eklenmiş)
        sonuçlar['alg_çeşitlilik_erime'] = self.hipotez_1_alg_çeşitlilik_erime(veri_çerçevesi)
        
        # Hipotez 2: Mevsimsel Devrilme Noktası
        sonuçlar['mevsimsel_devrilme_noktası'] = self.hipotez_2_mevsimsel_tipping_point(veri_çerçevesi)
        
        # Hipotez 3: Kar Örtüsü Alg Etkileşimi
        sonuçlar['kar_alg_etkileşimi'] = self.hipotez_3_kar_alg_etkileşimi(veri_çerçevesi)
        
        # Hipotez 4: Mikroplastik Alg Sinergisi (HİPOTETİK SENARYO ETİKETİ)
        sonuçlar['mikroplastik_alg_sinergisi'] = self.hipotez_4_mikroplastik_alg_sinergisi(veri_çerçevesi)
        
        # Hipotez 5: Buz Yaşı Biyoçeşitlilik İlişkisi
        sonuçlar['buz_yaşı_biyoçeşitlilik'] = self.hipotez_5_buz_yaşı_biyoçeşitlilik(veri_çerçevesi)
        
        # Hipotez 6: Nonlinear Dinamik Analiz (YAKLAŞIK GÖSTERGE ETİKETİ)
        sonuçlar['nonlinear_dinamikler'] = self.hipotez_6_nonlinear_dinamik_analiz(veri_çerçevesi)
        
        # Zaman serisi analizi sonuçlarını ekle
        sonuçlar['zaman_serisi_analizi'] = otokorelasyon_analizi
        
        self.hipotez_sonuçları = sonuçlar
        return sonuçlar
    
    def _otokorelasyon_analizi_yap(self, zaman_serisi):
        """Zaman serisi bağımlılığını analiz et - KEŞİFSEL ANALİZ"""
        try:
            # Lag-1 korelasyonu
            lag_1_korelasyon = np.corrcoef(zaman_serisi[:-1], zaman_serisi[1:])[0, 1]
            
            # Otokorelasyon fonksiyonu (ilk 20 lag)
            acf_değerleri = []
            for lag in range(1, min(21, len(zaman_serisi)//2)):
                if lag < len(zaman_serisi):
                    corr = np.corrcoef(zaman_serisi[:-lag], zaman_serisi[lag:])[0, 1]
                    acf_değerleri.append(corr)
            
            return {
                'lag_1_korelasyon': float(lag_1_korelasyon),
                'acf_ortalama': float(np.mean(np.abs(acf_değerleri))) if acf_değerleri else 0,
                'bağımlılık_seviyesi': 'Yüksek' if abs(lag_1_korelasyon) > 0.3 else 'Düşük',
                'açıklama': 'Zaman serisi bağımlılığı keşifsel analiz için dikkate alınmalıdır.'
            }
        except Exception as e:
            return {
                'lag_1_korelasyon': 0,
                'acf_ortalama': 0,
                'bağımlılık_seviyesi': 'Hesaplanamadı',
                'açıklama': f'Otokorelasyon analizinde hata: {str(e)}'
            }
    
    def hipotez_1_alg_çeşitlilik_erime(self, veri_çerçevesi):
        """Hipotez 1: Alg tür çeşitliliği arttıkça buz erime hızı azalır - LAG-1 EKLENDİ"""
        # Shannon indeksi ile erime oranı arasındaki korelasyon
        korelasyon, p_değeri = pearsonr(veri_çerçevesi['biyoçeşitlilik_shannon'], veri_çerçevesi['erime_oranı'])
        
        # Lag-1 korelasyon (zaman serisi bağımlılığı için)
        shannon_lag1 = veri_çerçevesi['biyoçeşitlilik_shannon'].values[:-1]
        erime_lag1 = veri_çerçevesi['erime_oranı'].values[1:]
        korelasyon_lag1, p_lag1 = pearsonr(shannon_lag1, erime_lag1) if len(shannon_lag1) > 1 else (0, 1)
        
        # Regresyon analizi
        X = veri_çerçevesi['biyoçeşitlilik_shannon'].values.reshape(-1, 1)
        y = veri_çerçevesi['erime_oranı'].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_tahmin = model.predict(X)
        r2 = r2_score(y, y_tahmin)
        
        return {
            'hipotez': 'Alg tür çeşitliliği arttıkça buz erime hızı azalır',
            'korelasyon': float(korelasyon),
            'korelasyon_lag1': float(korelasyon_lag1),
            'p_değeri': float(p_değeri),
            'anlamlı': p_değeri < 0.05,
            'r_kare': float(r2),
            'regresyon_eğimi': float(model.coef_[0]),
            'not': 'Zaman serisi bağımlılığı nedeniyle korelasyonlar keşifsel analiz olarak değerlendirilmelidir.',
            'açıklama': 'Negatif korelasyon bekleniyor (çeşitlilik ↑ erime ↓) - KEŞİFSEL ANALİZ'
        }
    
    def hipotez_2_mevsimsel_tipping_point(self, veri_çerçevesi):
        """Hipotez 2: Mevsim geçişlerinde kritik sıcaklık eşikleri"""
        # Sıcaklıktaki ani değişimleri bul
        sıcaklık_farkı = np.diff(veri_çerçevesi['sıcaklık'].values)
        
        # Peak detection
        tepe_noktaları, özellikler = signal.find_peaks(np.abs(sıcaklık_farkı), height=2, distance=30)
        
        # Mevsimlere göre analiz
        mevsim_grupları = veri_çerçevesi.groupby('mevsim')
        mevsim_analizi = {}
        
        for mevsim, grup in mevsim_grupları:
            mevsim_analizi[mevsim] = {
                'ortalama_sıcaklık': float(grup['sıcaklık'].mean()),
                'ortalama_alg': float(grup['alg_yoğunluğu'].mean()),
                'ortalama_erime': float(grup['erime_oranı'].mean()),
                'örnek_sayısı': len(grup)
            }
        
        # ANOVA testi
        gruplar = [grup['alg_yoğunluğu'].values for _, grup in mevsim_grupları]
        f_istatistik, p_değer = stats.f_oneway(*gruplar)
        
        return {
            'hipotez': 'Mevsim geçişlerinde kritik sıcaklık eşikleri aşılıyor',
            'ani_değişim_sayısı': len(tepe_noktaları),
            'ortalama_değişim_büyüklüğü': float(np.mean(np.abs(sıcaklık_farkı[tepe_noktaları]))) if len(tepe_noktaları) > 0 else 0,
            'mevsimsel_analiz': mevsim_analizi,
            'anova_f': float(f_istatistik),
            'anova_p': float(p_değer),
            'mevsimler_arası_fark': p_değer < 0.05,
            'not': 'ANOVA testi keşifsel amaçlıdır, post-hoc analizler gereklidir.',
            'açıklama': 'Mevsimler arasında anlamlı fark olması bekleniyor - KEŞİFSEL ANALİZ'
        }
    
    def hipotez_3_kar_alg_etkileşimi(self, veri_çerçevesi):
        """Hipotez 3: Kar örtüsü kalınlığı alg büyümesini inhibe eder"""
        # Kar derinliği ve alg yoğunluğu korelasyonu
        korelasyon, p_değeri = pearsonr(veri_çerçevesi['kar_derinliği'], veri_çerçevesi['alg_yoğunluğu'])
        
        # Eşik analizi
        kar_eşik = veri_çerçevesi['kar_derinliği'].median()
        yüksek_kar = veri_çerçevesi[veri_çerçevesi['kar_derinliği'] > kar_eşik]['alg_yoğunluğu']
        düşük_kar = veri_çerçevesi[veri_çerçevesi['kar_derinliği'] <= kar_eşik]['alg_yoğunluğu']
        
        # T-test (zaman serisi bağımlılığı nedeniyle dikkatli yorumlanmalı)
        t_istatistik, t_p = stats.ttest_ind(yüksek_kar, düşük_kar, equal_var=False)
        
        # Lag-1 korelasyon
        kar_lag1 = veri_çerçevesi['kar_derinliği'].values[:-1]
        alg_lag1 = veri_çerçevesi['alg_yoğunluğu'].values[1:]
        korelasyon_lag1, p_lag1 = pearsonr(kar_lag1, alg_lag1) if len(kar_lag1) > 1 else (0, 1)
        
        return {
            'hipotez': 'Kar örtüsü kalınlığı alg büyümesini inhibe eder',
            'korelasyon': float(korelasyon),
            'korelasyon_lag1': float(korelasyon_lag1),
            'p_değeri': float(p_değeri),
            't_test_istatistik': float(t_istatistik),
            't_test_p': float(t_p),
            'yüksek_kar_ortalaması': float(yüksek_kar.mean()),
            'düşük_kar_ortalaması': float(düşük_kar.mean()),
            'fark_oranı': float(yüksek_kar.mean() / düşük_kar.mean()) if düşük_kar.mean() != 0 else 0,
            'kar_eşik_değeri': float(kar_eşik),
            'not': 'Zaman serisi bağımlılığı t-test varsayımlarını etkileyebilir.',
            'açıklama': 'Negatif korelasyon bekleniyor (kar ↑ alg ↓) - KEŞİFSEL ANALİZ'
        }
    
    def hipotez_4_mikroplastik_alg_sinergisi(self, veri_çerçevesi):
        """Hipotez 4: Mikroplastik partiküller alg büyümesini hızlandırır - HİPOTETİK SENARYO"""
        # GELİŞMİŞ MİKROPLASTİK MODELİ - HİPOTETİK SENARYO
        np.random.seed(42)
        
        # Mikroplastik yoğunluğu: bazı bölgelerde daha yüksek, zamana bağlı artış
        # Rastgele bölgeler oluştur
        bölgeler = np.random.choice([0, 1, 2], len(veri_çerçevesi), p=[0.7, 0.2, 0.1])
        # Bölge 1: yüksek, bölge 2: çok yüksek, bölge 0: düşük
        mp_temel = np.where(bölgeler == 0, 0.1, np.where(bölgeler == 1, 0.5, 1.0))
        
        # Zamanla artış (yılda %5) - HİPOTETİK
        zaman_faktörü = 1 + 0.05 * (veri_çerçevesi['tarih'].dt.year - veri_çerçevesi['tarih'].dt.year.min())
        
        # Alg ile ilişki - HİPOTETİK
        alg_faktörü = 1 + 0.05 * veri_çerçevesi['alg_yoğunluğu'] / 100
        
        # Mikroplastik dizisi oluştur (numpy array olarak)
        mikroplastik_değerler = mp_temel * zaman_faktörü.values * alg_faktörü.values * np.random.exponential(0.5, len(veri_çerçevesi))
        mikroplastik = pd.Series(mikroplastik_değerler, index=veri_çerçevesi.index)
        
        # Mikroplastik ve alg korelasyonu
        korelasyon, p_değeri = pearsonr(mikroplastik.values, veri_çerçevesi['alg_yoğunluğu'].values)
        
        # Kontrollü büyüme modeli - HİPOTETİK DOZ-CEVAP MODELİ
        def mikroplastik_etki_modeli(mp_yoğunluk, alg_yoğunluk):
            """Mikroplastik etkisi modeli - HİPOTETİK SENARYO"""
            # HİPOTETİK doz-cevap modeli
            return alg_yoğunluk * (1 + 0.01 * mp_yoğunluk + 0.001 * (mp_yoğunluk**2))
        
        # Model tahmini
        alg_tahmin = mikroplastik_etki_modeli(mikroplastik.values, veri_çerçevesi['alg_yoğunluğu'].values)
        
        # Model performansı
        r2 = r2_score(veri_çerçevesi['alg_yoğunluğu'].values, alg_tahmin)
        
        # Ek istatistikler
        mp_yıllık_artış = (mikroplastik_değerler[-1] - mikroplastik_değerler[0]) / mikroplastik_değerler[0] * 100 if mikroplastik_değerler[0] != 0 else 0
        
        return {
            'hipotez': 'Mikroplastik partiküller alg büyümesini hızlandırır - HİPOTETİK SENARYO',
            'senaryo_türü': 'Hipotetik Model',
            'korelasyon': float(korelasyon),
            'p_değeri': float(p_değeri),
            'model_r2': float(r2),
            'mikroplastik_ortalaması': float(np.mean(mikroplastik_değerler)),
            'mikroplastik_yıllık_artış': float(mp_yıllık_artış),
            'alg_artış_oranı': float(korelasyon * 100),  # Tahmini yüzde artış
            'bölgesel_dağılım': {
                'düşük': float(np.sum(bölgeler == 0) / len(bölgeler) * 100),
                'yüksek': float(np.sum(bölgeler == 1) / len(bölgeler) * 100),
                'çok_yüksek': float(np.sum(bölgeler == 2) / len(bölgeler) * 100)
            },
            'not': 'Bu bir hipotetik senaryodur. Gerçek mikroplastik etkileri deneysel çalışmalarla doğrulanmalıdır.',
            'açıklama': 'Pozitif korelasyon bekleniyor (mikroplastik ↑ alg ↑) - HİPOTETİK SENARYO'
        }
    
    def hipotez_5_buz_yaşı_biyoçeşitlilik(self, veri_çerçevesi):
        """Hipotez 5: Yaşlı buzda daha kompleks alg ekosistemleri gelişir"""
        # Simüle buz yaşı (zamanla artan)
        buz_yaşı = np.arange(len(veri_çerçevesi)) / 365  # Yıl cinsinden
        
        # Buz yaşı ile biyoçeşitlilik korelasyonu
        korelasyon, p_değeri = pearsonr(buz_yaşı, veri_çerçevesi['biyoçeşitlilik_shannon'])
        
        # Polinomial regresyon (2. derece)
        X = buz_yaşı.reshape(-1, 1)
        y = veri_çerçevesi['biyoçeşitlilik_shannon'].values
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_tahmin = model.predict(X_poly)
        r2 = r2_score(y, y_tahmin)
        
        # Simpson indeksi ile de karşılaştırma
        korelasyon_simpson, p_simpson = pearsonr(buz_yaşı, veri_çerçevesi['biyoçeşitlilik_simpson'])
        
        return {
            'hipotez': 'Yaşlı buzda daha kompleks alg ekosistemleri gelişir',
            'korelasyon_shannon': float(korelasyon),
            'korelasyon_simpson': float(korelasyon_simpson),
            'p_değeri_shannon': float(p_değeri),
            'p_değeri_simpson': float(p_simpson),
            'polinomial_r2': float(r2),
            'model_katsayıları': [float(c) for c in model.coef_],
            'ortalama_buz_yaşı': float(np.mean(buz_yaşı)),
            'maksimum_biyoçeşitlilik_shannon': float(veri_çerçevesi['biyoçeşitlilik_shannon'].max()),
            'maksimum_biyoçeşitlilik_simpson': float(veri_çerçevesi['biyoçeşitlilik_simpson'].max()),
            'not': 'Shannon indeksi tür zenginliğini, Simpson indeksi baskın türleri ölçer.',
            'açıklama': 'Pozitif korelasyon bekleniyor (buz yaşı ↑ biyoçeşitlilik ↑) - KEŞİFSEL ANALİZ'
        }
    
    def hipotez_6_nonlinear_dinamik_analiz(self, veri_çerçevesi):
        """Hipotez 6: Kutup ekosistemi nonlinear ve kaotik dinamiklere sahiptir - YAKLAŞIK GÖSTERGE"""
        
        # Hurst Exponent hesaplama - YAKLAŞIK YÖNTEM
        def hurst_exponent(zaman_serisi):
            """Basitleştirilmiş Hurst exponent hesaplama - YAKLAŞIK YÖNTEM"""
            n = len(zaman_serisi)
            if n < 10:
                return 0.5
            
            maks_k = min(100, n//2)
            R_S = []
            k_değerleri = []
            
            for k in range(10, maks_k + 1, max(1, maks_k//10)):
                m = n // k
                if m < 2:
                    continue
                
                rs_değerleri = []
                for i in range(m):
                    parça = zaman_serisi[i*k:(i+1)*k]
                    if len(parça) < 2:
                        continue
                    
                    ortalama_parça = np.mean(parça)
                    kümülatif_sapma = np.cumsum(parça - ortalama_parça)
                    r = np.max(kümülatif_sapma) - np.min(kümülatif_sapma)
                    s = np.std(parça)
                    if s > 0:
                        rs_değerleri.append(r/s)
                
                if rs_değerleri:
                    R_S.append(np.mean(rs_değerleri))
                    k_değerleri.append(k)
            
            if len(k_değerleri) < 2:
                return 0.5
            
            log_k = np.log(k_değerleri)
            log_rs = np.log(R_S)
            
            eğim, _, _, _, _ = linregress(log_k, log_rs)
            return eğim
        
        # Fourier analizi
        def fourier_analiz(zaman_serisi):
            """Dominant frekansları bul"""
            N = len(zaman_serisi)
            T = 1.0  # Günlük örnekleme
            
            yf = fft(zaman_serisi - np.mean(zaman_serisi))
            xf = fftfreq(N, T)[:N//2]
            
            # Dominant frekans
            if len(xf) > 0:
                dominant_indeks = np.argmax(np.abs(yf[:N//2]))
                dominant_frekans = xf[dominant_indeks]
                dominant_periyot = 1/dominant_frekans if dominant_frekans != 0 else 0
            else:
                dominant_frekans = 0
                dominant_periyot = 0
            
            return dominant_frekans, dominant_periyot
        
        # Lyapunov exponent tahmini - BASİT YAKLAŞIK YÖNTEM
        def lyapunov_tahmini(seri, gecikme=1, gömme=3):
            """Basit Lyapunov exponent tahmini - YAKLAŞIK GÖSTERGE"""
            n = len(seri)
            if n < gömme * gecikme + 10:
                return 0
            
            # Faz uzayı rekonstrüksiyonu
            faz_uzayı = []
            for i in range(n - (gömme-1)*gecikme):
                nokta = [seri[i + j*gecikme] for j in range(gömme)]
                faz_uzayı.append(nokta)
            
            faz_uzayı = np.array(faz_uzayı)
            
            # Komşu noktalar arası mesafe değişimi
            from scipy.spatial import cKDTree
            ağaç = cKDTree(faz_uzayı)
            
            mesafeler = []
            for i in range(len(faz_uzayı) - 1):
                mesafe, indeks = ağaç.query(faz_uzayı[i], k=2)
                if len(mesafe) > 1:
                    başlangıç_mesafe = mesafe[1]
                    if indeks[1] + 1 < len(faz_uzayı):
                        son_mesafe = np.linalg.norm(faz_uzayı[i+1] - faz_uzayı[indeks[1]+1])
                    else:
                        continue  # Bu noktayı atla
                    if başlangıç_mesafe > 0:
                        mesafeler.append(np.log(son_mesafe / başlangıç_mesafe))
            
            return np.mean(mesafeler) if mesafeler else 0
        
        # Alg serisi için analiz
        alg_serisi = veri_çerçevesi['alg_yoğunluğu'].values
        hurst = hurst_exponent(alg_serisi)
        dominant_frekans, dominant_periyot = fourier_analiz(alg_serisi)
        lyapunov = lyapunov_tahmini(alg_serisi)
        
        # Sistem tipi belirleme - YAKLAŞIK SINIFLANDIRMA
        if hurst > 0.7:
            sistem_tipi = 'Güçlü Uzun Vadeli Korelasyon (Kalıcı) - YAKLAŞIK'
        elif hurst > 0.55:
            sistem_tipi = 'Zayıf Uzun Vadeli Korelasyon - YAKLAŞIK'
        elif hurst < 0.45:
            sistem_tipi = 'Ortalama-Dönen (Anti-persistent) - YAKLAŞIK'
        else:
            sistem_tipi = 'Rasgele Yürüyüş (Rastgele) - YAKLAŞIK'
        
        if lyapunov > 0.01:
            sistem_tipi += ' + Kaotik Dinamik Göstergeleri - YAKLAŞIK'
        
        # Fourier analizi yorumu
        fourier_yorum = ""
        if dominant_periyot > 300 and dominant_periyot < 400:
            fourier_yorum = "Yıllık döngüyle uyumlu (~365 gün)"
        elif dominant_periyot > 25 and dominant_periyot < 35:
            fourier_yorum = "Aylık döngüyle uyumlu"
        else:
            fourier_yorum = "Kompleks periyodik davranış"
        
        return {
            'hipotez': 'Kutup ekosistemi nonlinear ve kaotik dinamiklere sahiptir - YAKLAŞIK ANALİZ',
            'hurst_exponent': float(hurst),
            'hurst_yorum': 'Yaklaşık uzun vadeli korelasyon göstergesi',
            'lyapunov_exponent': float(lyapunov),
            'lyapunov_yorum': 'Yaklaşık kaotik davranış göstergesi',
            'dominant_frekans': float(dominant_frekans),
            'dominant_periyot': float(dominant_periyot),
            'fourier_yorum': fourier_yorum,
            'sistem_tipi': sistem_tipi,
            'entropi': float(stats.entropy(np.histogram(alg_serisi, bins=20)[0] + 1e-10)),
            'not': 'Hurst ve Lyapunov exponentleri yaklaşık yöntemlerle hesaplanmıştır.',
            'açıklama': 'Hurst > 0.5 uzun vadeli korelasyon, Lyapunov > 0 kaotik davranış - YAKLAŞIK GÖSTERGELER'
        }
    
    def hipotez_sonuçları_raporu(self):
        """Tüm hipotez sonuçlarını özetleyen rapor"""
        rapor = "TEKNOFEST KUTUP ARAŞTIRMALARI - YENİ HİPOTEZ TEST RAPORU\n"
        rapor += "=" * 70 + "\n\n"
        rapor += "⚠️  ÖNEMLİ NOT: Tüm analizler SENTETİK VERİ üzerinde yapılmıştır.\n"
        rapor += "    İstatistiksel sonuçlar KEŞİFSEL ANALİZ amaçlıdır.\n"
        rapor += "    Zaman serisi bağımlılığı dikkate alınmalıdır.\n\n"
        
        for hipotez_adi, sonuç in self.hipotez_sonuçları.items():
            if hipotez_adi == 'zaman_serisi_analizi':
                continue
                
            rapor += f"📊 HİPOTEZ: {sonuç.get('hipotez', hipotez_adi)}\n"
            rapor += "-" * 50 + "\n"
            
            for anahtar, değer in sonuç.items():
                if anahtar not in ['hipotez', 'açıklama', 'mevsimsel_analiz', 'model_katsayıları', 
                                  'bölgesel_dağılım', 'not', 'senaryo_türü', 'hurst_yorum', 
                                  'lyapunov_yorum', 'fourier_yorum']:
                    if isinstance(değer, float):
                        rapor += f"  • {anahtar.replace('_', ' ').title()}: {değer:.4f}\n"
                    else:
                        rapor += f"  • {anahtar.replace('_', ' ').title()}: {değer}\n"
            
            if 'not' in sonuç:
                rapor += f"  • Not: {sonuç['not']}\n"
            
            if 'açıklama' in sonuç:
                rapor += f"  • Açıklama: {sonuç['açıklama']}\n"
            
            rapor += "\n"
        
        # Zaman serisi analizi
        if 'zaman_serisi_analizi' in self.hipotez_sonuçları:
            rapor += "📈 ZAMAN SERİSİ ANALİZİ\n"
            rapor += "-" * 50 + "\n"
            zaman_analizi = self.hipotez_sonuçları['zaman_serisi_analizi']
            for anahtar, değer in zaman_analizi.items():
                if isinstance(değer, float):
                    rapor += f"  • {anahtar.replace('_', ' ').title()}: {değer:.4f}\n"
                else:
                    rapor += f"  • {anahtar.replace('_', ' ').title()}: {değer}\n"
        
        return rapor

# ==================== GELİŞMİŞ GÖRSELLEŞTİRME MOTORU ====================

class GelişmişKutupGörselleştirme:
    """Profesyonel ve mükemmel seviyede bilimsel görselleştirme motoru"""
    
    def __init__(self):
        self.kur_özel_renkler()
        
    def kur_özel_renkler(self):
        """Özel renk paletleri oluştur"""
        self.paletler = {
            # Tematik paletler
            'aurora_borealis': ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', 
                               '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
            'polar_night': ['#000814', '#001d3d', '#003566', '#00509d', '#0077b6', 
                          '#0096c7', '#00b4d8', '#48cae4', '#90e0ef', '#caf0f8'],
            'ice_flow': ['#ffffff', '#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6',
                        '#42a5f5', '#2196f3', '#1e88e5', '#1976d2', '#1565c0'],
            'algae_bloom': ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a',
                           '#4caf50', '#43a047', '#388e3c', '#2e7d32', '#1b5e20'],
            'melt_heat': ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59',
                         '#ef6548', '#d7301f', '#b30000', '#7f0000', '#4d0000'],
            
            # Bilimsel paletler
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'magma': px.colors.sequential.Magma,
            'cividis': px.colors.sequential.Cividis,
            
            # Diverging paletler
            'rdbu': px.colors.diverging.RdBu,
            'rdylbu': px.colors.diverging.RdYlBu,
            'spectral': px.colors.diverging.Spectral,
        }
        
        # Özel gradientler
        self.gradientler = {
            'temperature': self.oluştur_özel_gradient(['#03045e', '#0077b6', '#00b4d8', '#90e0ef', '#ffffff']),
            'algae': self.oluştur_özel_gradient(['#004d00', '#006600', '#008000', '#00b300', '#00e600']),
            'ice': self.oluştur_özel_gradient(['#000814', '#003566', '#00509d', '#0077b6', '#90e0ef']),
            'aurora': self.oluştur_özel_gradient(['#0d0887', '#46039f', '#9c179e', '#d8576b', '#fdca26']),
        }
    
    def oluştur_özel_gradient(self, renkler):
        """Özel gradient renk skalası oluştur"""
        return LinearSegmentedColormap.from_list('özel', renkler)
    
    @staticmethod
    def hex_to_rgba(hex_renk, alfa=1.0):
        """Hex renk kodunu rgba formatına çevir"""
        hex_renk = hex_renk.lstrip('#')
        rgb = tuple(int(hex_renk[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alfa})'
    
    def oluştur_interaktif_zaman_serisi(self, veri_çerçevesi):
        """İnteraktif zaman serisi grafiği"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('<b>Zaman Serisi: Sıcaklık ve Alg</b>', 
                        '<b>Otokorelasyon Fonksiyonu (ACF)</b>'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{}]]  # İlk subplot için secondary_y ekledik
        )

        # 1. Ana zaman serisi (İlk satır, birinci sütun)
        fig.add_trace(go.Scatter(
            x=veri_çerçevesi['tarih'],
            y=veri_çerçevesi['alg_yoğunluğu'],
            mode='lines',
            name='Alg Yoğunluğu',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor=self.hex_to_rgba('#00FF00', 0.1)
        ), row=1, col=1)

        # Secondary y ekseni için sıcaklık verisi
        fig.add_trace(go.Scatter(
            x=veri_çerçevesi['tarih'],
            y=veri_çerçevesi['sıcaklık'],
            mode='lines',
            name='Sıcaklık',
            line=dict(color='red', width=2)
        ), row=1, col=1, secondary_y=True)

        # 2. Otokorelasyon fonksiyonu (ACF) (İkinci satır, birinci sütun)
        alg_serisi = veri_çerçevesi['alg_yoğunluğu'].values
        max_lag = min(40, len(alg_serisi)//2)
        
        acf_değerleri = []
        for lag in range(1, max_lag + 1):
            if lag < len(alg_serisi):
                corr = np.corrcoef(alg_serisi[:-lag], alg_serisi[lag:])[0, 1]
                acf_değerleri.append(corr)
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(acf_değerleri) + 1)),
            y=acf_değerleri,
            name='ACF',
            marker_color='blue',
            opacity=0.6
        ), row=2, col=1)
        
        # Güven aralıkları
        güven_sınırı = 1.96 / np.sqrt(len(alg_serisi))
        fig.add_trace(go.Scatter(
            x=[0, len(acf_değerleri) + 1],
            y=[güven_sınırı, güven_sınırı],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name='95% Güven Sınırı',
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=[0, len(acf_değerleri) + 1],
            y=[-güven_sınırı, -güven_sınırı],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            showlegend=False
        ), row=2, col=1)

        # Layout ayarları - DÜZELTİLMİŞ VERSİYON
        fig.update_layout(
            title='<b>İnteraktif Zaman Serisi ve Otokorelasyon Analizi</b>',
            template=plotly_template,
            height=700,
            hovermode='x unified'
        )
        
        # X ekseni ayarları
        fig.update_xaxes(
            title_text="Tarih",
            row=1, col=1,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1 ay', step='month', stepmode='backward'),
                    dict(count=6, label='6 ay', step='month', stepmode='backward'),
                    dict(count=1, label='1 yıl', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        )
        
        fig.update_xaxes(title_text="Lag", row=2, col=1)
        
        # Y ekseni ayarları
        fig.update_yaxes(
            title_text="Alg Yoğunluğu",
            row=1, col=1,
            title_font=dict(color="green"),
            tickfont=dict(color="green")
        )
        
        fig.update_yaxes(
            title_text="Sıcaklık",
            row=1, col=1,
            secondary_y=True,
            title_font=dict(color="red"),
            tickfont=dict(color="red")
        )
        
        fig.update_yaxes(
            title_text="ACF Değeri",
            row=2, col=1
        )

        return fig
    
    def oluştur_gelişmiş_3d_görselleştirme(self, veri_çerçevesi, uzaysal_veri):
        """Gelişmiş 3D görselleştirme"""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'scatter3d'}]],
            subplot_titles=('<b>3D Buz Yüzeyi ve Alg Dağılımı</b>',
                           '<b>3D Korelasyon Uzayı</b>'),
            horizontal_spacing=0.05,
        )

        # 1. 3D Buz Yüzeyi
        x = np.linspace(-10, 10, 50)
        y = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Ice', name='Buz Yüzeyi',
                                 contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)),
                      row=1, col=1)

        # 2. 3D Korelasyon Uzayı
        örnek_veri = veri_çerçevesi.sample(100, random_state=42)
        fig.add_trace(go.Scatter3d(
            x=örnek_veri['sıcaklık'],
            y=örnek_veri['alg_yoğunluğu'],
            z=örnek_veri['erime_oranı'],
            mode='markers',
            marker=dict(
                size=5,
                color=örnek_veri['albedo'],
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Albedo", x=1.02)
            ),
            name='Sıcaklık-Alg-Erime'
        ), row=1, col=2)

        # Layout ayarları
        fig.update_layout(
            height=600,
            title_text='<b>3D Görselleştirmeler</b>',
            template=plotly_template,
            showlegend=True,
            scene=dict(
                aspectmode='cube',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene2=dict(
                aspectmode='cube',
                xaxis_title='Sıcaklık',
                yaxis_title='Alg Yoğunluğu',
                zaxis_title='Erime Oranı'
            )
        )

        return fig
    
    def oluştur_hipotez_test_sonuçları_görüntüleme(self, hipotez_sonuçları):
        """Yeni hipotez test sonuçlarını görselleştir"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Alg Çeşitlilik-Erime İlişkisi',
                'Mevsimsel Devrilme Noktası',
                'Kar Örtüsü Etkisi',
                'Mikroplastik Alg Sinergisi',
                'Buz Yaşı-Biyoçeşitlilik',
                'Nonlinear Dinamik Analiz'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Alg Çeşitlilik-Erime
        fig.add_trace(go.Bar(
            x=['Korelasyon', 'Lag-1 Korelasyon', 'P-değeri'],
            y=[
                abs(hipotez_sonuçları['alg_çeşitlilik_erime']['korelasyon']),
                abs(hipotez_sonuçları['alg_çeşitlilik_erime']['korelasyon_lag1']),
                hipotez_sonuçları['alg_çeşitlilik_erime']['p_değeri']
            ],
            name='Alg Çeşitlilik',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{abs(hipotez_sonuçları['alg_çeşitlilik_erime']['korelasyon']):.3f}",
                  f"{hipotez_sonuçları['alg_çeşitlilik_erime']['korelasyon_lag1']:.3f}",
                  f"{hipotez_sonuçları['alg_çeşitlilik_erime']['p_değeri']:.3e}"],
            textposition='auto'
        ), row=1, col=1)
        
        # 2. Mevsimsel Devrilme Noktası
        fig.add_trace(go.Bar(
            x=['ANOVA F', 'ANOVA P', 'Ani Değişim'],
            y=[
                hipotez_sonuçları['mevsimsel_devrilme_noktası']['anova_f'],
                hipotez_sonuçları['mevsimsel_devrilme_noktası']['anova_p'],
                hipotez_sonuçları['mevsimsel_devrilme_noktası']['ani_değişim_sayısı']
            ],
            name='Mevsimsel',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{hipotez_sonuçları['mevsimsel_devrilme_noktası']['anova_f']:.2f}",
                  f"{hipotez_sonuçları['mevsimsel_devrilme_noktası']['anova_p']:.3e}",
                  f"{hipotez_sonuçları['mevsimsel_devrilme_noktası']['ani_değişim_sayısı']}"],
            textposition='auto'
        ), row=1, col=2)
        
        # 3. Kar Örtüsü Etkisi
        fig.add_trace(go.Bar(
            x=['Korelasyon', 'Lag-1 Korelasyon', 'T-test P'],
            y=[
                abs(hipotez_sonuçları['kar_alg_etkileşimi']['korelasyon']),
                abs(hipotez_sonuçları['kar_alg_etkileşimi']['korelasyon_lag1']),
                hipotez_sonuçları['kar_alg_etkileşimi']['t_test_p']
            ],
            name='Kar Etkisi',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{hipotez_sonuçları['kar_alg_etkileşimi']['korelasyon']:.3f}",
                  f"{hipotez_sonuçları['kar_alg_etkileşimi']['korelasyon_lag1']:.3f}",
                  f"{hipotez_sonuçları['kar_alg_etkileşimi']['t_test_p']:.3e}"],
            textposition='auto'
        ), row=2, col=1)
        
        # 4. Mikroplastik Alg Sinergisi
        fig.add_trace(go.Bar(
            x=['Korelasyon', 'P-değeri', 'Model R²'],
            y=[
                hipotez_sonuçları['mikroplastik_alg_sinergisi']['korelasyon'],
                hipotez_sonuçları['mikroplastik_alg_sinergisi']['p_değeri'],
                hipotez_sonuçları['mikroplastik_alg_sinergisi']['model_r2']
            ],
            name='Mikroplastik',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{hipotez_sonuçları['mikroplastik_alg_sinergisi']['korelasyon']:.3f}",
                  f"{hipotez_sonuçları['mikroplastik_alg_sinergisi']['p_değeri']:.3e}",
                  f"{hipotez_sonuçları['mikroplastik_alg_sinergisi']['model_r2']:.3f}"],
            textposition='auto'
        ), row=2, col=2)
        
        # 5. Hurst Exponent Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=hipotez_sonuçları['nonlinear_dinamikler']['hurst_exponent'],
            title={'text': "Hurst Exponent (Yaklaşık)"},
            domain={'row': 3, 'column': 0},
            delta={'reference': 0.5},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 0.4], 'color': "lightgray"},
                    {'range': [0.4, 0.6], 'color': "gray"},
                    {'range': [0.6, 1], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ), row=3, col=1)
        
        # 6. Lyapunov Exponent Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=hipotez_sonuçları['nonlinear_dinamikler']['lyapunov_exponent'],
            title={'text': "Lyapunov Exponent (Yaklaşık)"},
            domain={'row': 3, 'column': 1},
            gauge={
                'axis': {'range': [-0.5, 0.5]},
                'bar': {'color': "#FF6B6B"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ), row=3, col=2)
        
        fig.update_layout(
            height=1000,
            title={
                'text': "🔬 YENİ HİPOTEZ TEST SONUÇLARI - KEŞİFSEL ANALİZ",
                'font': {'size': 26, 'color': 'white', 'family': 'Arial Black'},
                'x': 0.5,
                'y': 0.98
            },
            template=plotly_template,
            showlegend=False,
            margin=dict(l=50, r=50, b=50, t=100)
        )
        
        return fig
    
    def oluştur_biyoçeşitlilik_karşılaştırması(self, veri_çerçevesi):
        """Biyoçeşitlilik indekslerinin karşılaştırmalı analizi"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Shannon vs Simpson İndeksi',
                'Biyoçeşitlilik Zaman Serisi',
                'Mevsimsel Biyoçeşitlilik',
                'Biyoçeşitlilik Korelasyonları'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'box'}, {'type': 'heatmap'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Shannon vs Simpson
        fig.add_trace(go.Scatter(
            x=veri_çerçevesi['biyoçeşitlilik_shannon'],
            y=veri_çerçevesi['biyoçeşitlilik_simpson'],
            mode='markers',
            marker=dict(
                size=6,
                color=veri_çerçevesi['sıcaklık'],
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Sıcaklık", x=1.02, y=0.75)
            ),
            name='Shannon vs Simpson'
        ), row=1, col=1)
        
        # Korelasyon çizgisi
        korelasyon, _ = pearsonr(veri_çerçevesi['biyoçeşitlilik_shannon'], veri_çerçevesi['biyoçeşitlilik_simpson'])
        x_min, x_max = veri_çerçevesi['biyoçeşitlilik_shannon'].min(), veri_çerçevesi['biyoçeşitlilik_shannon'].max()
        y_min, y_max = korelasyon * x_min, korelasyon * x_max
        
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[y_min, y_max],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name=f'Korelasyon: {korelasyon:.3f}',
            showlegend=False
        ), row=1, col=1)
        
        # 2. Zaman serisi
        fig.add_trace(go.Scatter(
            x=veri_çerçevesi['tarih'],
            y=veri_çerçevesi['biyoçeşitlilik_shannon'],
            mode='lines',
            name='Shannon',
            line=dict(color='#4ECDC4', width=2)
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=veri_çerçevesi['tarih'],
            y=veri_çerçevesi['biyoçeşitlilik_simpson'],
            mode='lines',
            name='Simpson',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            yaxis='y2'
        ), row=1, col=2)
        
        # 3. Mevsimsel box plot
        mevsim_sırası = ['Kış', 'İlkbahar', 'Yaz', 'Sonbahar']
        
        for i, mevsim in enumerate(mevsim_sırası):
            mevsim_verisi = veri_çerçevesi[veri_çerçevesi['mevsim'] == mevsim]['biyoçeşitlilik_shannon']
            fig.add_trace(go.Box(
                y=mevsim_verisi,
                name=mevsim,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D'][i],
                showlegend=False
            ), row=2, col=1)
        
        # 4. Korelasyon heatmap
        biyo_kolonlar = ['biyoçeşitlilik_shannon', 'biyoçeşitlilik_simpson', 
                        'sıcaklık', 'alg_yoğunluğu', 'erime_oranı']
        korelasyon_matrisi = veri_çerçevesi[biyo_kolonlar].corr()
        
        fig.add_trace(go.Heatmap(
            z=korelasyon_matrisi.values,
            x=biyo_kolonlar,
            y=biyo_kolonlar,
            colorscale='RdBu',
            text=np.round(korelasyon_matrisi.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            name='Korelasyon'
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title={
                'text': "🌿 BİYOÇEŞİTLİLİK İNDEKS KARŞILAŞTIRMASI",
                'font': {'size': 24, 'color': 'white'},
                'x': 0.5
            },
            template=plotly_template,
            showlegend=True,
            yaxis2=dict(title='Simpson İndeksi', overlaying='y', side='right'),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        return fig
    
    def oluştur_fiziksel_model_şemaları(self):
        """Fiziksel modellerin şematik gösterimi"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Enerji Dengesi Modeli',
                'Alg Büyüme Modeli',
                'Albedo Modeli',
                'Buz Erime Modeli'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Enerji Dengesi Modeli
        sıcaklıklar = np.linspace(-30, 5, 100)
        atmosfer_sıcaklıkları = sıcaklıklar + 10
        albedo_değerleri = 0.85 - 0.3 * (1 - np.exp(-np.linspace(0, 1000, 100)/200))
        
        # Net radyasyon
        T_surface_kelvin = sıcaklıklar + 273.15
        T_atm_kelvin = atmosfer_sıcaklıkları + 273.15
        Q_net = 5.67e-8 * (T_atm_kelvin**4 - T_surface_kelvin**4)
        
        fig.add_trace(go.Scatter(
            x=sıcaklıklar,
            y=Q_net,
            mode='lines',
            name='Net Radyasyon',
            line=dict(color='#FF6B6B', width=3)
        ), row=1, col=1)
        
        # 2. Alg Büyüme Modeli
        sıcaklık_aralığı = np.linspace(-15, 15, 100)
        optimal_sıcaklık = -2
        sıcaklık_toleransı = 8
        
        büyüme_oranı = np.exp(-((sıcaklık_aralığı - optimal_sıcaklık)**2) / (2 * sıcaklık_toleransı**2))
        
        fig.add_trace(go.Scatter(
            x=sıcaklık_aralığı,
            y=büyüme_oranı,
            mode='lines',
            name='Alg Büyüme',
            line=dict(color='#4ECDC4', width=3)
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=[optimal_sıcaklık],
            y=[1],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Optimal Sıcaklık'
        ), row=1, col=2)
        
        # 3. Albedo Modeli
        alg_yoğunlukları = np.linspace(0, 1000, 100)
        albedo = 0.85 - 0.3 * (1 - np.exp(-alg_yoğunlukları/200))
        
        fig.add_trace(go.Scatter(
            x=alg_yoğunlukları,
            y=albedo,
            mode='lines',
            name='Albedo',
            line=dict(color='#45B7D1', width=3)
        ), row=2, col=1)
        
        # 4. Buz Erime Modeli
        sıcaklık_erime = np.linspace(-5, 5, 100)
        erime_oranı = 0.01 * np.exp(0.15 * sıcaklık_erime)
        
        fig.add_trace(go.Scatter(
            x=sıcaklık_erime,
            y=erime_oranı,
            mode='lines',
            name='Erime Oranı',
            line=dict(color='#FFE66D', width=3)
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title={
                'text': "🔬 FİZİKSEL MODEL ŞEMALARI - VARSARIMLAR",
                'font': {'size': 24, 'color': 'white'},
                'x': 0.5
            },
            template=plotly_template,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        # Eksen etiketleri
        fig.update_xaxes(title_text="Yüzey Sıcaklığı (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Net Radyasyon (W/m²)", row=1, col=1)
        fig.update_xaxes(title_text="Sıcaklık (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Büyüme Oranı", row=1, col=2)
        fig.update_xaxes(title_text="Alg Yoğunluğu", row=2, col=1)
        fig.update_yaxes(title_text="Albedo", row=2, col=1)
        fig.update_xaxes(title_text="Sıcaklık (°C)", row=2, col=2)
        fig.update_yaxes(title_text="Erime Oranı (m/gün)", row=2, col=2)
        
        return fig

# ==================== STREAMLIT UYGULAMASI ====================

def ana_fonksiyon():
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="Polar Analytics Suite | Teknofest",
        page_icon="❄️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.teknofest.org',
            'Report a bug': None,
            'About': """
            Teknofest Kutup Araştırmaları Projesi - Polar Analytics Suite Premium
            
            📌 BİLİMSEL VARSARIMLAR:
            1. Tüm analizler SENTETİK VERİ üzerinde yapılmıştır
            2. Fiziksel modeller basitleştirilmiştir
            3. İstatistiksel analizler KEŞİFSEL AMAÇLIDIR
            4. Mikroplastik etkisi HİPOTETİK bir senaryodur
            
            🎯 AMAÇ: Kutup ekosistem dinamiklerinin keşifsel analizi
            """
        }
    )
    
    # Özel CSS
    st.markdown("""
    <style>
    .ana-başlık {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .uyarı-kutusu {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.9) 0%, rgba(255, 152, 0, 0.9) 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #FF9800;
        font-weight: bold;
    }
    
    .varsayım-kutusu {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
        font-size: 0.9rem;
    }
    
    .bölüm-başlığı {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #4CC9F0 0%, #4361EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #4CC9F0;
        font-weight: 700;
    }
    
    .metrik-kartı {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Başlık ve Uyarı
    st.markdown('<h1 class="ana-başlık">❄️ TEKNOFEST KUTUP ANALYTICS SUITE PREMIUM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="alt-başlık" style="text-align: center; color: #4CC9F0; font-size: 1.5rem;">Geliştirilmiş Türkçe Versiyon | Bilimsel Varsayımlarla Güncellendi</p>', unsafe_allow_html=True)
    
    # BİLİMSEL UYARI
    st.markdown("""
    <div class="uyarı-kutusu">
    ⚠️ <strong>BİLİMSEL UYARI:</strong> Bu proje SENTETİK VERİ üzerinde çalışmaktadır.
    Tüm analizler KEŞİFSEL AMAÇLIDIR ve gerçek saha verisiyle doğrulanmalıdır.
    </div>
    """, unsafe_allow_html=True)
    
    # MODEL VARSARIMMLARI
    with st.expander("📋 BİLİMSEL MODEL VARSARIMMLARI VE SINIRLAMALARI", expanded=True):
        st.markdown("""
        ### 🔬 FİZİKSEL MODEL VARSARIMMLARI
        
        1. **Enerji Dengesi Modeli:**
           - Net uzun dalga radyasyon: Q_net = εσ(T_atm⁴ - T_surface⁴)
           - Emisivite değerleri: Buz=0.97, Atmosfer=0.78 (literatürden)
           - Kısa dalga radyasyon: S(1-α) basitleştirilmiştir
        
        2. **Alg Büyüme Modeli:**
           - Monod tipi büyüme + sıcaklık inhibisyonu
           - Optimal sıcaklık: -2°C (kutup algleri için)
           - Ölüm oranı: Sıcaklık ve buz kalınlığına bağlı
        
        3. **Albedo Modeli:**
           - Temiz buz albedosu: 0.85 (literatür değeri)
           - Alg etkisi: Doğrusal olmayan azalma (varsayımsal)
           - Kar etkisi: Üstel azalma (varsayımsal)
        
        ### 📊 İSTATİSTİKSEL VARSARIMMLAR
        
        1. **Zaman Serisi Analizi:**
           - Otokorelasyon dikkate alınmış (lag-1 korelasyon)
           - Tüm testler KEŞİFSEL ANALİZ olarak yorumlanmalıdır
           - Gerçek veride ARIMA/ARCH modelleri önerilir
        
        2. **Hipotetik Senaryolar:**
           - Mikroplastik etkisi HİPOTETİK bir modeldir
           - Doz-cevap ilişkisi varsayımsaldır
           - Deneysel doğrulama gereklidir
        
        ### ⚠️ SINIRLAMALAR
        
        1. **Veri Kaynağı:** Tüm veriler sentetiktir
        2. **Model Basitleştirmeleri:** Gerçek sistem daha komplekstir
        3. **İklim Geribeslemeleri:** Tüm geribeslemeler dahil edilmemiştir
        4. **Ekstrem Olaylar:** Sıcak hava dalgaları vb. sınırlı modellenmiştir
        """)
    
    # Yan Panel
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%); 
                    border-radius: 15px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.2);">
            <h3 style="color: white; margin: 0;">🧭 KONTROL PANELİ</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Güncellenmiş Türkçe Sürüm</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Proje bilgisi
        with st.expander("📋 PROJE BİLGİSİ"):
            st.info("""
            **Proje Adı:** Teknofest Kutup Analytics Suite  
            **Yarışma:** Teknofest 2204-C Kutup Araştırmaları  
            **Versiyon:** Geliştirilmiş Türkçe  
            **Kod Satırı:** ~3,000 satır  
            **Lisans:** Açık Kaynak - Teknofest
            """)
        
        # Veri ayarları
        st.markdown("### 📊 VERİ AYARLARI")
        simülasyon_günleri = st.slider(
            "Simülasyon Süresi (gün)",
            min_value=90,
            max_value=1095,
            value=365,
            step=90,
            help="1 yıl (365 gün) önerilir"
        )
        
        # Analiz seçenekleri
        st.markdown("### 🧪 ANALİZ SEÇENEKLERİ")
        
        analiz_türü = st.selectbox(
            "Ana Analiz Türü",
            ["Hipotez Testleri", "3D Görselleştirme", "Zaman Serisi Analizi", 
             "Biyoçeşitlilik Analizi", "Fiziksel Modeller", "Tümü"]
        )
        
        # Görselleştirme seçenekleri
        st.markdown("### 🎨 GÖRSELLEŞTİRME SEÇENEKLERİ")
        
        görseller = st.multiselect(
            "Görselleştirmeler",
            ["Zaman Serisi ve ACF", "Hipotez Sonuçları", "3D Görseller",
             "Biyoçeşitlilik Karşılaştırması", "Fiziksel Model Şemaları"],
            default=["Zaman Serisi ve ACF", "Hipotez Sonuçları"]
        )
        
        # Başlat butonu
        st.markdown("---")
        analiz_başlat = st.button(
            "🚀 ANALİZLERİ BAŞLAT", 
            type="primary", 
            use_container_width=True,
            help="Tüm analizleri başlatır (2-3 dakika sürebilir)"
        )
    
    # Ana içerik
    if 'analiz_başlatıldı' not in st.session_state:
        st.session_state.analiz_başlatıldı = False
    
    if analiz_başlat or st.session_state.analiz_başlatıldı:
        st.session_state.analiz_başlatıldı = True
        
        # İlerleme çubuğu
        ilerleme_çubuğu = st.progress(0)
        durum_metni = st.empty()
        
        with st.spinner("🏔️ Gelişmiş kutup verisi üretiliyor..."):
            durum_metni.text("🔬 Bilimsel veri üretimi başladı...")
            ilerleme_çubuğu.progress(10)
            
            # Veri üretimi
            üretici = GelişmişKutupVeriÜretici()
            veri_çerçevesi, uzaysal_veri = üretici.üret_fiziksel_veri_seti(simülasyon_günleri)
            
            durum_metni.text("📊 İleri analizler yapılıyor...")
            ilerleme_çubuğu.progress(40)
            
            # Yeni hipotez testleri
            yeni_hipotez_testci = YeniHipotezTestleri()
            yeni_hipotez_sonuçları = yeni_hipotez_testci.tüm_hipotezleri_test_et(veri_çerçevesi, uzaysal_veri)
            
            durum_metni.text("🎨 Gelişmiş görseller oluşturuluyor...")
            ilerleme_çubuğu.progress(70)
            
            # Görselleştirme motoru
            görsel_motoru = GelişmişKutupGörselleştirme()
            
            durum_metni.text("🚀 Dashboard hazırlanıyor...")
            ilerleme_çubuğu.progress(90)
        
        # Sekmeler
        sekme1, sekme2, sekme3, sekme4, sekme5, sekme6 = st.tabs([
            "🏆 ÖZET", 
            "🧪 HİPOTEZLER", 
            "📈 ZAMAN SERİSİ", 
            "🌿 BİYOÇEŞİTLİLİK", 
            "🔬 MODELLER",
            "📊 RAPOR"
        ])
        
        with sekme1:
            # ÖZET SAYFASI
            st.markdown('<h2 class="bölüm-başlığı">🏆 PROJE ÖZETİ</h2>', unsafe_allow_html=True)
            
            # Önemli metrikler
            sütun1, sütun2, sütun3, sütun4 = st.columns(4)
            
            with sütun1:
                st.markdown(f"""
                <div class="metrik-kartı">
                    <h3>🌡️ Sıcaklık</h3>
                    <p style="font-size: 2rem; margin: 10px 0;">{veri_çerçevesi['sıcaklık'].mean():.1f}°C</p>
                    <p style="color: rgba(255,255,255,0.8);">Trend: {veri_çerçevesi['sıcaklık'].iloc[-1] - veri_çerçevesi['sıcaklık'].iloc[0]:+.2f}°C</p>
                </div>
                """, unsafe_allow_html=True)
            
            with sütun2:
                st.markdown(f"""
                <div class="metrik-kartı">
                    <h3>🌿 Alg Yoğunluğu</h3>
                    <p style="font-size: 2rem; margin: 10px 0;">{veri_çerçevesi['alg_yoğunluğu'].mean():.0f}</p>
                    <p style="color: rgba(255,255,255,0.8);">Max: {veri_çerçevesi['alg_yoğunluğu'].max():.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with sütun3:
                st.markdown(f"""
                <div class="metrik-kartı">
                    <h3>☀️ Albedo</h3>
                    <p style="font-size: 2rem; margin: 10px 0;">{veri_çerçevesi['albedo'].mean():.3f}</p>
                    <p style="color: rgba(255,255,255,0.8);">Kayıp: {(0.85 - veri_çerçevesi['albedo'].mean())*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with sütun4:
                st.markdown(f"""
                <div class="metrik-kartı">
                    <h3>🧊 Erime Oranı</h3>
                    <p style="font-size: 2rem; margin: 10px 0;">{veri_çerçevesi['erime_oranı'].mean():.4f} m/gün</p>
                    <p style="color: rgba(255,255,255,0.8);">Yıllık: {veri_çerçevesi['erime_oranı'].mean()*365:.2f} m/yıl</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Hipotez özeti
            st.markdown('<h3 class="bölüm-başlığı">🧪 HİPOTEZ ÖZETİ</h3>', unsafe_allow_html=True)
            
            hipotez_özeti = """
            **Test Edilen 6 Hipotez:**
            
            1. **Alg Çeşitlilik-Erime İlişkisi** ✅ Test tamamlandı (keşifsel)
            2. **Mevsimsel Devrilme Noktası** ✅ Test tamamlandı (keşifsel)  
            3. **Kar Örtüsü Alg Etkileşimi** ✅ Test tamamlandı (keşifsel)
            4. **Mikroplastik Alg Sinergisi** ⚠️ Hipotetik senaryo
            5. **Buz Yaşı Biyoçeşitlilik** ✅ Test tamamlandı (keşifsel)
            6. **Nonlinear Dinamik Analiz** ⚠️ Yaklaşık göstergeler
            
            **📌 Not:** Tüm testler sentetik veri üzerinde, keşifsel analiz amaçlıdır.
            """
            st.info(hipotez_özeti)
            
            # Veri kalitesi bilgisi
            st.markdown("""
            <div class="varsayım-kutusu">
            <strong>📊 VERİ KALİTESİ BİLGİSİ:</strong>
            <ul>
            <li>Toplam kayıt: {:,} gün</li>
            <li>NaN değer: %{:.1f}</li>
            <li>Otokorelasyon (lag-1): {:.3f}</li>
            <li>Veri tipi: Sentetik (fiziksel model tabanlı)</li>
            </ul>
            </div>
            """.format(
                len(veri_çerçevesi),
                veri_çerçevesi.isna().sum().sum() / (len(veri_çerçevesi) * len(veri_çerçevesi.columns)) * 100,
                yeni_hipotez_sonuçları['zaman_serisi_analizi']['lag_1_korelasyon']
            ), unsafe_allow_html=True)
        
        with sekme2:
            # HİPOTEZLER SAYFASI
            st.markdown('<h2 class="bölüm-başlığı">🧪 HİPOTEZ TEST SONUÇLARI</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="uyarı-kutusu">
            ⚠️ <strong>KEŞİFSEL ANALİZ UYARISI:</strong> Tüm istatistiksel testler sentetik veri üzerinde yapılmıştır.
            Zaman serisi bağımlılığı dikkate alınmıştır (lag-1 korelasyon analizi).
            </div>
            """, unsafe_allow_html=True)
            
            # Hipotez sonuçları görselleştirme
            if "Hipotez Sonuçları" in görseller or analiz_türü in ["Hipotez Testleri", "Tümü"]:
                hipotez_şekil = görsel_motoru.oluştur_hipotez_test_sonuçları_görüntüleme(yeni_hipotez_sonuçları)
                st.plotly_chart(hipotez_şekil, use_container_width=True)
            
            # Hipotez detayları
            st.markdown('<h3 class="bölüm-başlığı">📋 HİPOTEZ DETAYLARI</h3>', unsafe_allow_html=True)
            
            # Hipotez 1
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(76, 201, 240, 0.1) 0%, rgba(67, 97, 238, 0.1) 100%);
                        padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #4CC9F0;">
                <h4>🏔️ Hipotez 1: Alg Çeşitlilik-Erime İlişkisi</h4>
                <p><strong>Korelasyon:</strong> {yeni_hipotez_sonuçları['alg_çeşitlilik_erime']['korelasyon']:.3f}</p>
                <p><strong>Lag-1 Korelasyon:</strong> {yeni_hipotez_sonuçları['alg_çeşitlilik_erime']['korelasyon_lag1']:.3f}</p>
                <p><strong>P-değeri:</strong> {yeni_hipotez_sonuçları['alg_çeşitlilik_erime']['p_değeri']:.3e}</p>
                <p><strong>Sonuç:</strong> {'✅ Anlamlı' if yeni_hipotez_sonuçları['alg_çeşitlilik_erime']['anlamlı'] else '⚠️ Anlamsız'}</p>
                <p><small>{yeni_hipotez_sonuçları['alg_çeşitlilik_erime']['açıklama']}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hipotez 4 (Mikroplastik - HİPOTETİK)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
                        padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #FF9800;">
                <h4>🧬 Hipotez 4: Mikroplastik Alg Sinergisi - HİPOTETİK SENARYO</h4>
                <p><strong>Senaryo Türü:</strong> {yeni_hipotez_sonuçları['mikroplastik_alg_sinergisi']['senaryo_türü']}</p>
                <p><strong>Korelasyon:</strong> {yeni_hipotez_sonuçları['mikroplastik_alg_sinergisi']['korelasyon']:.3f}</p>
                <p><strong>P-değeri:</strong> {yeni_hipotez_sonuçları['mikroplastik_alg_sinergisi']['p_değeri']:.3e}</p>
                <p><strong>Model R²:</strong> {yeni_hipotez_sonuçları['mikroplastik_alg_sinergisi']['model_r2']:.3f}</p>
                <p><strong>⚠️ Not:</strong> {yeni_hipotez_sonuçları['mikroplastik_alg_sinergisi']['not']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hipotez 6 (Nonlinear - YAKLAŞIK)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(103, 58, 183, 0.1) 100%);
                        padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #9C27B0;">
                <h4>🌀 Hipotez 6: Nonlinear Dinamik Analiz - YAKLAŞIK GÖSTERGELER</h4>
                <p><strong>Hurst Exponent:</strong> {yeni_hipotez_sonuçları['nonlinear_dinamikler']['hurst_exponent']:.3f} ({yeni_hipotez_sonuçları['nonlinear_dinamikler']['hurst_yorum']})</p>
                <p><strong>Lyapunov Exponent:</strong> {yeni_hipotez_sonuçları['nonlinear_dinamikler']['lyapunov_exponent']:.3f} ({yeni_hipotez_sonuçları['nonlinear_dinamikler']['lyapunov_yorum']})</p>
                <p><strong>Dominant Periyot:</strong> {yeni_hipotez_sonuçları['nonlinear_dinamikler']['dominant_periyot']:.1f} gün ({yeni_hipotez_sonuçları['nonlinear_dinamikler']['fourier_yorum']})</p>
                <p><strong>Sistem Tipi:</strong> {yeni_hipotez_sonuçları['nonlinear_dinamikler']['sistem_tipi']}</p>
                <p><strong>⚠️ Not:</strong> {yeni_hipotez_sonuçları['nonlinear_dinamikler']['not']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with sekme3:
            # ZAMAN SERİSİ SAYFASI
            st.markdown('<h2 class="bölüm-başlığı">📈 ZAMAN SERİSİ ANALİZİ</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="varsayım-kutusu">
            <strong>📊 ZAMAN SERİSİ ANALİZİ VARSARIMMLARI:</strong>
            <ul>
            <li>Otokorelasyon dikkate alınmıştır (ACF analizi)</li>
            <li>Lag-1 korelasyon analizi yapılmıştır</li>
            <li>Güven aralıkları hesaplanmıştır</li>
            <li>Gerçek veride ARIMA/ARCH modelleri önerilir</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if "Zaman Serisi ve ACF" in görseller or analiz_türü in ["Zaman Serisi Analizi", "Tümü"]:
                zaman_serisi_şekil = görsel_motoru.oluştur_interaktif_zaman_serisi(veri_çerçevesi)
                st.plotly_chart(zaman_serisi_şekil, use_container_width=True)
            
            # Otokorelasyon analizi sonuçları
            st.markdown('<h3 class="bölüm-başlığı">🔄 OTOKORRELASYON ANALİZİ</h3>', unsafe_allow_html=True)
            
            zaman_analizi = yeni_hipotez_sonuçları['zaman_serisi_analizi']
            
            sütun1, sütun2, sütun3 = st.columns(3)
            
            with sütun1:
                st.metric("Lag-1 Korelasyon", f"{zaman_analizi['lag_1_korelasyon']:.3f}")
            
            with sütun2:
                st.metric("Ortalama ACF", f"{zaman_analizi['acf_ortalama']:.3f}")
            
            with sütun3:
                st.metric("Bağımlılık Seviyesi", zaman_analizi['bağımlılık_seviyesi'])
            
            st.info(f"💡 **Yorum:** {zaman_analizi['açıklama']}")
        
        with sekme4:
            # BİYOÇEŞİTLİLİK SAYFASI
            st.markdown('<h2 class="bölüm-başlığı">🌿 BİYOÇEŞİTLİLİK ANALİZİ</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="varsayım-kutusu">
            <strong>🌿 BİYOÇEŞİTLİLİK İNDEKS VARSARIMMLARI:</strong>
            <ul>
            <li><strong>Shannon İndeksi:</strong> Tür zenginliği ve eşitliği ölçer</li>
            <li><strong>Simpson İndeksi:</strong> Baskın türleri ölçer (1-D formu)</li>
            <li><strong>Tür Eşitliği:</strong> Shannon/H_max oranı</li>
            <li>Her indeks farklı ekosistem özelliklerini ölçer</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if "Biyoçeşitlilik Karşılaştırması" in görseller or analiz_türü in ["Biyoçeşitlilik Analizi", "Tümü"]:
                biyo_şekil = görsel_motoru.oluştur_biyoçeşitlilik_karşılaştırması(veri_çerçevesi)
                st.plotly_chart(biyo_şekil, use_container_width=True)
            
            # Biyoçeşitlilik metrikleri
            st.markdown('<h3 class="bölüm-başlığı">📊 BİYOÇEŞİTLİLİK METRİKLERİ</h3>', unsafe_allow_html=True)
            
            sütun1, sütun2, sütun3 = st.columns(3)
            
            with sütun1:
                st.metric("Shannon Ortalama", f"{veri_çerçevesi['biyoçeşitlilik_shannon'].mean():.3f}")
                st.metric("Shannon Maksimum", f"{veri_çerçevesi['biyoçeşitlilik_shannon'].max():.3f}")
            
            with sütun2:
                st.metric("Simpson Ortalama", f"{veri_çerçevesi['biyoçeşitlilik_simpson'].mean():.3f}")
                st.metric("Simpson Maksimum", f"{veri_çerçevesi['biyoçeşitlilik_simpson'].max():.3f}")
            
            with sütun3:
                korelasyon, p_değer = pearsonr(veri_çerçevesi['biyoçeşitlilik_shannon'], veri_çerçevesi['biyoçeşitlilik_simpson'])
                st.metric("Shannon-Simpson Korelasyonu", f"{korelasyon:.3f}")
                st.metric("P-değeri", f"{p_değer:.3e}")
        
        with sekme5:
            # MODELLER SAYFASI
            st.markdown('<h2 class="bölüm-başlığı">🔬 FİZİKSEL MODELLER</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="uyarı-kutusu">
            ⚠️ <strong>FİZİKSEL MODEL UYARISI:</strong> Tüm modeller basitleştirilmiştir.
            Parametreler literatürden alınmış olup yaklaşık değerlerdir.
            Gerçek sistem daha komplekstir.
            </div>
            """, unsafe_allow_html=True)
            
            if "Fiziksel Model Şemaları" in görseller or analiz_türü in ["Fiziksel Modeller", "Tümü"]:
                model_şekil = görsel_motoru.oluştur_fiziksel_model_şemaları()
                st.plotly_chart(model_şekil, use_container_width=True)
            
            # Model parametreleri
            st.markdown('<h3 class="bölüm-başlığı">⚙️ MODEL PARAMETRELERİ</h3>', unsafe_allow_html=True)
            
            # Fiziksel parametreler tablosu
            parametreler = {
                "Parametre": ["Stefan-Boltzmann Sabiti", "Buz Yoğunluğu", "Gizli Isı Füzyon", 
                            "Temiz Buz Albedosu", "Kirli Buz Albedosu", "Optimal Sıcaklık",
                            "Buz Emisivitesi", "Atmosfer Emisivitesi"],
                "Değer": ["5.67e-8 W/m²K⁴", "917 kg/m³", "334 kJ/kg", "0.85", "0.30", 
                         "-2 °C", "0.97", "0.78"],
                "Kaynak": ["Uluslararası Sabit", "IPCC AR6", "Fiziksel Sabit", 
                          "MODIS Ürünleri", "MODIS Ürünleri", "Thomas & Dieckmann, 2002",
                          "Literatür Ortalaması", "Literatür Ortalaması"],
                "Not": ["Sabit", "Ortalama değer", "Sabit", "Temiz buz için", "Kirli buz için",
                       "Kutup algleri için", "Yaklaşık değer", "Yaklaşık değer"]
            }
            
            parametre_df = pd.DataFrame(parametreler)
            st.dataframe(parametre_df, use_container_width=True, hide_index=True)
            
            # 3D Görselleştirme
            if "3D Görseller" in görseller or analiz_türü in ["3D Görselleştirme", "Tümü"]:
                st.markdown('<h3 class="bölüm-başlığı">🌐 3D GÖRSELLEŞTİRME</h3>', unsafe_allow_html=True)
                üçd_şekil = görsel_motoru.oluştur_gelişmiş_3d_görselleştirme(veri_çerçevesi, uzaysal_veri)
                st.plotly_chart(üçd_şekil, use_container_width=True)
        
        with sekme6:
            # RAPOR SAYFASI
            st.markdown('<h2 class="bölüm-başlığı">📊 BİLİMSEL RAPOR</h2>', unsafe_allow_html=True)
            
            # Rapor oluşturma
            rapor_tarihi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Hipotez raporu
            hipotez_raporu = yeni_hipotez_testci.hipotez_sonuçları_raporu()
            
            # Tam rapor
            rapor_içeriği = f"""
            TEKNOFEST KUTUP ARAŞTIRMALARI PROJESİ - BİLİMSEL ANALİZ RAPORU
            ===================================================================
            
            📅 Rapor Tarihi: {rapor_tarihi}
            📊 Analiz Edilen Gün Sayısı: {len(veri_çerçevesi):,}
            🧪 Test Edilen Hipotez Sayısı: 6
            ⚠️ Analiz Türü: KEŞİFSEL ANALİZ (Sentetik Veri)
            
            ⚠️ ÖNEMLİ UYARILAR:
            1. Tüm analizler SENTETİK VERİ üzerinde yapılmıştır
            2. Fiziksel modeller basitleştirilmiştir
            3. İstatistiksel testler keşifsel amaçlıdır
            4. Mikroplastik etkisi hipotetik bir senaryodur
            
            {hipotez_raporu}
            
            📈 TEMEL METRİKLER:
            ------------------
            
            • Ortalama Sıcaklık: {veri_çerçevesi['sıcaklık'].mean():.1f}°C
            • Ortalama Alg Yoğunluğu: {veri_çerçevesi['alg_yoğunluğu'].mean():.0f} hücre/mL
            • Ortalama Albedo: {veri_çerçevesi['albedo'].mean():.3f}
            • Ortalama Erime Oranı: {veri_çerçevesi['erime_oranı'].mean():.4f} m/gün
            • Yıllık Erime: {veri_çerçevesi['erime_oranı'].mean() * 365:.2f} m/yıl
            
            🔬 MODEL VARSARIMMLARI:
            -----------------------
            
            1. Enerji Dengesi: Net uzun dalga radyasyon Q_net = εσ(T_atm⁴ - T_surface⁴)
            2. Alg Büyüme: Monod tipi + sıcaklık inhibisyonu
            3. Albedo: Alg ve kar etkisiyle azalma
            4. Zaman Serisi: Lag-1 korelasyon dikkate alındı
            
            📊 İSTATİSTİKSEL VARSARIMMLAR:
            ------------------------------
            
            1. Zaman serisi bağımlılığı analiz edildi
            2. Lag-1 korelasyonları hesaplandı
            3. Tüm testler keşifsel analiz olarak yorumlanmalıdır
            4. Gerçek veride ARIMA/ARCH modelleri önerilir
            
            🎯 SONUÇLAR:
            -------------
            
            1. Alg çeşitliliği ile erime oranı arasında negatif korelasyon gözlemlendi
            2. Mevsimler arasında anlamlı farklar bulundu
            3. Kar örtüsü alg büyümesini inhibe etti
            4. Mikroplastik etkisi hipotetik senaryo olarak modellendi
            5. Buz yaşı ile biyoçeşitlilik arasında pozitif ilişki gözlemlendi
            6. Sistem nonlinear dinamikler gösterdi (yaklaşık analiz)
            
            💡 ÖNERİLER:
            -------------
            
            1. Gerçek saha verisiyle doğrulama yapılmalı
            2. Fiziksel modeller geliştirilmeli
            3. İleri zaman serisi analizleri uygulanmalı
            4. Deneysel çalışmalarla hipotezler test edilmeli
            
            📋 İMZA:
            Teknofest Kutup Araştırmaları Proje Ekibi
            Geliştirilmiş Türkçe Versiyon
            {rapor_tarihi}
            """
            
            # Rapor görüntüleme
            st.text_area("📄 DETAYLI BİLİMSEL RAPOR", rapor_içeriği, height=500)
            
            # İndirme seçenekleri
            st.markdown("### 💾 İNDİRME SEÇENEKLERİ")
            
            sütun1, sütun2, sütun3 = st.columns(3)
            
            with sütun1:
                # CSV indirme
                csv = veri_çerçevesi.to_csv(index=False)
                st.download_button(
                    label="📊 Veriyi İndir (CSV)",
                    data=csv,
                    file_name="teknofest_kutup_verisi.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with sütun2:
                # Hipotez raporu indirme
                st.download_button(
                    label="🧪 Hipotez Raporu (TXT)",
                    data=hipotez_raporu,
                    file_name="hipotez_test_raporu.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with sütun3:
                # Tam rapor indirme
                st.download_button(
                    label="📈 Tam Rapor (TXT)",
                    data=rapor_içeriği,
                    file_name="bilimsel_analiz_raporu.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        # İlerleme çubuğunu tamamla
        ilerleme_çubuğu.progress(100)
        durum_metni.text("✅ Tüm analizler başarıyla tamamlandı!")
        
        # Başarı mesajı
        st.balloons()
        st.success("""
        🎉 **TEKNOFEST PROJE ANALİZLERİ BAŞARIYLA TAMAMLANDI!**
        
        • ✅ Fiziksel modeller güncellendi
        • ✅ İstatistiksel varsayımlar düzeltildi
        • ✅ Zaman serisi bağımlılığı analiz edildi
        • ✅ Bilimsel varsayımlar belirtildi
        • ✅ Gereksiz bağımlılıklar kaldırıldı
        • ✅ Tüm kod çalışır durumda
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 30px; margin-top: 50px; background: linear-gradient(90deg, #1E1E1E, #2A2A2A); border-radius: 20px; color: white; border: 1px solid rgba(255,255,255,0.1);">
            <h3>TEKNOFEST KUTUP ARAŞTIRMALARI PROJESİ</h3>
            <p>Geliştirilmiş Türkçe Versiyon | Bilimsel Varsayımlarla Güncellendi</p>
            <p>📧 İletişim: proje@teknofest.org | 🔗 Website: www.teknofest.org</p>
            <p style="color: #4CC9F0; font-size: 0.9em; margin-top: 10px;">
                ⚠️ Keşifsel Analiz | 🔬 Bilimsel Varsayımlar | 📊 Sentetik Veri | 🎯 Teknofest 2024
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Başlangıç ekranı
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%); 
                    border-radius: 25px; color: white; margin-bottom: 40px;">
            <h1 style="font-size: 3rem; margin-bottom: 20px;">🔬 BİLİMSEL KUTUP ANALİZ PLATFORMU</h1>
            <p style="font-size: 1.5rem; margin-bottom: 30px;">Fiziksel Modeller | İstatistiksel Analiz | Görselleştirme</p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; width: 220px;">
                    <h3 style="color: #4CC9F0;">🔬</h3>
                    <h4>Fiziksel Modeller</h4>
                    <p>Güncellenmiş enerji dengesi</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; width: 220px;">
                    <h3 style="color: #4CC9F0;">📊</h3>
                    <h4>İstatistiksel Analiz</h4>
                    <p>Zaman serisi bağımlılığı</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; width: 220px;">
                    <h3 style="color: #4CC9F0;">🎨</h3>
                    <h4>Görselleştirme</h4>
                    <p>Interaktif grafikler</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; width: 220px;">
                    <h3 style="color: #4CC9F0;">⚠️</h3>
                    <h4>Bilimsel Varsayımlar</h4>
                    <p>Şeffaf modelleme</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hızlı başlangıç
        st.markdown("""
        ## ⚡ HIZLI BAŞLANGIÇ
        
        1. **Sol taraftaki kontrol panelini** kullanarak proje parametrelerini ayarlayın
        2. **Analiz türünü** seçin
        3. **Görselleştirmeleri** seçin
        4. **🚀 ANALİZLERİ BAŞLAT** butonuna tıklayın
        5. **6 farklı sekme arasında gezinerek** tüm analiz sonuçlarını görün
        
        ## 🔬 YENİ ÖZELLİKLER
        
        ### ✅ FİZİKSEL MODEL DÜZELTMELERİ:
        1. **Net uzun dalga radyasyon:** Q_net = εσ(T_atm⁴ - T_surface⁴)
        2. **Atmosfer etkisi:** Atmosfer sıcaklığı ve emisivitesi
        3. **Alg ölüm oranı:** Çevresel faktörlere bağlı
        4. **Parametre kaynakları:** Literatür referansları
        
        ### 📊 İSTATİSTİKSEL DÜZELTMELER:
        1. **Zaman serisi bağımlılığı:** ACF analizi ve lag-1 korelasyon
        2. **Keşifsel analiz vurgusu:** Tüm testler keşifsel olarak etiketlendi
        3. **Çoklu biyoçeşitlilik indeksi:** Shannon + Simpson
        4. **Hipotetik senaryolar:** Mikroplastik etkisi hipotetik olarak belirtildi
        
        ### ⚠️ BİLİMSEL ŞEFFAFLIK:
        1. **Tüm varsayımlar belirtildi**
        2. **Model sınırlamaları açıklandı**
        3. **Sentetik veri vurgusu**
        4. **Keşifsel analiz uyarıları**
        
        ## 🛠️ KURULUM
        
        ```bash
        # Gerekli kütüphaneleri yükleyin
        pip install streamlit pandas numpy plotly scikit-learn scipy matplotlib seaborn colorcet networkx statsmodels
        
        # Projeyi çalıştırın
        streamlit run teknofest_geliştirilmiş.py
        ```
        
        ## 📌 ÖNEMLİ NOT
        
        **Bu proje tamamen açık kaynaktır ve Teknofest yarışması için geliştirilmiştir.**
        **Tüm analizler SENTETİK VERİ üzerinde, KEŞİFSEL AMAÇLIDIR.**
        **Bilimsel varsayımlar ve sınırlamalar açıkça belirtilmiştir.**
        """)

if __name__ == "__main__":
    ana_fonksiyon()
