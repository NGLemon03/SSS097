# leverage.py
class LeverageEngine:
    def __init__(self, ltv=0.6, maint=1.30, target=1.50, annual_rate=0.05):
        self.ltv, self.maint, self.target = ltv, maint, target
        self.daily = annual_rate / 365
        self.loan = 0.0
        self.int_paid = 0.0

    def avail(self, mkt_val):  # 可再借額度
        return max(self.ltv * mkt_val - self.loan, 0)

    def borrow(self, amt):  # 借錢
        self.loan += amt

    def repay(self, amt):  # 還錢
        self.loan = max(self.loan - amt, 0)

    def accrue(self):  # 每日計息
        i = self.loan * self.daily
        self.int_paid += i
        return i

    def margin_call(self, mkt_val):  # 強平所需賣出市值
        if mkt_val == 0:
            return 0
        ratio = (mkt_val - self.loan) / mkt_val
        if ratio >= self.maint:
            return 0
        x = max(0, mkt_val - self.loan / (1 - self.target))
        return min(x, mkt_val)