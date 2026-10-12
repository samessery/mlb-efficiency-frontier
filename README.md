# How Efficiently Do MLB Teams Turn Payroll into Wins?

Stochastic frontier analysis (SFA) and data envelopment analysis (DEA) of team payroll efficiency, 1,128 team-seasons, 1985 to 2023, with a luck-only simulation to separate real inefficiency from noise.

This extends two college projects (a fixed-effects study of payroll and winning, and a study of payroll dispersion and competitive balance). Both found that payroll buys wins at a stable rate. This repo asks the follow-up question: **given payroll, how far below the best-practice frontier do teams fall, and is that gap persistent, or does it rotate?**

## Main findings

1. **Payroll has diminishing returns.** Frontier elasticity of win % with respect to relative payroll is 0.125 (SE 0.011): doubling payroll relative to the league raises best-practice win % by about 9%.
2. **Luck alone cannot explain the gaps.** If every team were equally efficient and wins were pure binomial luck around the payroll-implied probability, the average efficiency score would be 0.936 (SFA) and the dispersion of scores 0.030. Observed: 0.873 and 0.068. The frontier's estimated one-sided noise term (sigma_u) is 0.177 observed vs 0.085 under the null (95th percentile 0.100).
3. **Inefficiency rotates.** Efficiency is correlated across seasons for 2 to 3 years (lag-1 autocorrelation 0.36, lag-2 0.20, lag-3 0.08, all outside the luck-only band), then falls to zero by year 4 (lag-4 0.01, lag-5 -0.03, both inside the band). Only about 5% of the variance in efficiency is a permanent team characteristic. This matches the "rotating inefficiency" idea in the sports economics literature (Hakes and Sauer): a team that beats its payroll tends to keep doing so for a couple of seasons, but contention windows open and close.
4. **A handful of teams are persistently different.** With robust standard errors that account for the autocorrelation above, 7 of 30 franchises have a long-run gap that is distinguishable from luck (30 tests, so one or two false positives are expected):

   | Beat payroll | Fell short of payroll |
   |---|---|
   | Oakland (+6.9% wins vs payroll-implied), St. Louis (+5.3%), New York Yankees (+4.4%, borderline) | Detroit (-8.4%), Kansas City (-6.6%), Baltimore (-6.6%, borderline), Colorado (-5.8%) |

   Tampa Bay (+5.6%) and Atlanta (+4.8%) look strong but their intervals include zero. Full table: `results/team_efficiency_table.csv`.
5. **Payroll explains less of winning since 2017.** The standard deviation of wins unexplained by payroll was 0.131 in 1985-2016 and rose to 0.169 in 2017-23 (excluding the 60-game 2020 season), with 2018 and 2019 (0.189, 0.193) above every earlier season and 2021-23 near the top of the earlier range (0.10 to 0.17). Binomial luck is constant across these years, so the increase is real, not noise. Possible reasons (not tested here) include teams deliberately tanking, and super-teams versus rebuilders.

## Method

| Step | What it does |
|---|---|
| Input / output | Payroll divided by the league average that season (removes inflation) and win percentage |
| SFA | `ln(win%) = a + b ln(relative payroll) + v - u`, with `v` normal noise and `u` half-normal inefficiency, estimated by maximum likelihood (own implementation). Efficiency = Battese-Coelli `E[exp(-u)|v-u]` |
| DEA | Output-oriented, variable returns to scale, season by season. A closed-form concave-envelope version is checked against the linear program |
| Luck-only null | 300 simulated leagues in which every team has the payroll-implied win probability and wins are binomial. Both estimators are rerun on each |
| Team table | Average log win % above the payroll-implied level, with a luck-only interval and an autocorrelation-robust (HAC) interval |

```
python/frontier.py       estimators, simulation, tables, figures
tests/test_frontier.py   recovers known parameters on simulated data; checks DEA LP vs closed form
data/                    team-season panel (wins, games, payroll)
results/                 all tables, simulation draws, run log
figures/                 frontier overview, team residuals, unexplained variation by season
```

```bash
pip install -r requirements.txt
pytest -q                      # 3 tests
python python/frontier.py      # about 1 minute
```

## Limits (read these before quoting any number)

- **"Efficiency" here means everything payroll and luck do not explain.** It bundles front-office skill with other things: player development and cheap pre-arbitration talent, ballpark, injuries beyond ordinary luck, and the fact that payroll in a given year reflects contracts signed earlier. I call it efficiency because that is the standard term, not because I can isolate management quality.
- **The frontier is a benchmark, not a target.** Win percentage is zero-sum (the league average is always 0.500), so average efficiency cannot be 1.
- **The null assumes a smooth payroll-to-wins curve and independent binomial luck.** If real luck has extra structure (for example runs scoring in clusters), the null is a little too tight. The estimated noise term is 0.091 against pure binomial noise of about 0.079, so the gap is small but present.
- **Payroll is nominal payroll converted to a share of the league average,** not player-level production. Salary data are team totals.
- **Era comparisons use few seasons** (5 to 18 per era), and the 2017-23 increase is descriptive.
- **Team labels** use current franchise names for relocated or renamed teams.
- DEA scores run lower than SFA scores because DEA treats the best noisy season as the frontier.

## Author

Samuel Essery, B.S. Quantitative Economics, Providence College.
