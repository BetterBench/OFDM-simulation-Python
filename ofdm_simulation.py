import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy

K = 64 # OFDM子载波数量
CP = K//4  #25%的循环前缀长度
P = 8  # 导频数
pilotValue = 3+3j  # 导频格式
Modulation_type = 'QAM16' #调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
channel_type ='random' # 信道类型，可选awgn
SNRdb = 25  # 接收端的信噪比（dB）
allCarriers = np.arange(K)  # 子载波编号 ([0, 1, ... K-1])
pilotCarrier = allCarriers[::K//P]  # 每间隔P个子载波一个导频
# 为了方便信道估计，将最后一个子载波也作为导频
pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])
P = P+1 # 导频的数量也需要加1
m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map[Modulation_type]
dataCarriers = np.delete(allCarriers, pilotCarriers)
payloadBits_per_OFDM = len(dataCarriers)*mu  # 每个 OFDM 符号的有效载荷位数
# 定义制调制方式
def Modulation(bits):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        symbol = PSK4.modulate(bits)
        return symbol
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        symbol = QAM64.modulate(bits)
        return symbol
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        symbol = QAM16.modulate(bits)
        return symbol
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        symbol = PSK8.modulate(bits)
        return symbol
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        symbol = BPSK.modulate(bits)
        return symbol
# 定义解调方式
def DeModulation(symbol):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        bits = PSK4.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        bits = QAM64.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        bits = QAM16.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        bits = PSK8.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        bits = BPSK.demodulate(symbol, demod_type='hard')
        return bits
# 定义信道
def add_awgn(x_s, snrDB):
    data_pwr = np.mean(abs(x_s**2))
    noise_pwr = data_pwr/(10**(snrDB/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j *
                            np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
    return x_s + noise, noise_pwr
def channel(in_signal, SNRdb, channel_type="awgn"):
    channelResponse = np.array([1, 0, 0.3+0.3j])  # 随意仿真信道冲击响应
    if channel_type == "random":
        convolved = np.convolve(in_signal, channelResponse)
        out_signal, noise_pwr = add_awgn(convolved, SNRdb)
    elif channel_type == "awgn":
        out_signal, noise_pwr = add_awgn(in_signal, SNRdb)
    return out_signal, noise_pwr
# 插入导频和数据，生成OFDM符号
def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)  # 子载波位置
    symbol[pilotCarriers] = pilotValue  # 在导频位置插入导频
    symbol[dataCarriers] = QAM_payload  # 在数据位置插入数据
    return symbol
# 快速傅里叶逆变换
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
# 添加循环前缀
def addCP(OFDM_time):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])
# 接收端，去除循环前缀
def removeCP(signal):
    return signal[CP:(CP+K)]

# 快速傅里叶变换
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
# 信道估计
def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # 取导频处的数据
    Hest_at_pilots = pilots / pilotValue  # LS信道估计s
    # 在导频载波之间进行插值以获得估计，然后利用插值估计得到数据下标处的信道响应
    Hest_abs = interpolate.interp1d(pilotCarriers, abs(
        Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = interpolate.interp1d(pilotCarriers, np.angle(
        Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    return Hest
# 均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def OFDM_simulation():
    # 产生比特流
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    # 比特信号调制
    QAM_s = Modulation(bits)
    # 生成OFDM符号
    OFDM_data = OFDM_symbol(QAM_s)
    # 快速逆傅里叶变换
    OFDM_time = IDFT(OFDM_data)
    # 添加循环前缀
    OFDM_withCP = addCP(OFDM_time)

    # 经过信道
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, SNRdb, "random")[0]

    # 去除循环前缀
    OFDM_RX_noCP = removeCP(OFDM_RX)
    # 快速傅里叶变换
    OFDM_demod = DFT(OFDM_RX_noCP)
    # 信道估计
    Hest = channelEstimate(OFDM_demod)
    # 均衡
    equalized_Hest = equalize(OFDM_demod, Hest)
    # 获取数据位置的数据
    def get_payload(equalized):
        return equalized[dataCarriers]
    QAM_est = get_payload(equalized_Hest)
    # 反映射，解调
    bits_est = DeModulation(QAM_est)
    # print(bits_est)
    print("误比特率BER： ", np.sum(abs(bits-bits_est))/len(bits))
if __name__ == '__main__':
    OFDM_simulation()
