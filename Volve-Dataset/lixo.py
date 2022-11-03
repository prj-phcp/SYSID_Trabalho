from scipy import stats

print(stats.uniform().rvs(10, random_state=None) >= 0.5)