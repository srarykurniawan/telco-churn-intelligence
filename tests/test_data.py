import pandas as pd

from telco.data.generator import CUSTOMER_COLUMNS, generate_customers, generate_monthly_kpis
from telco.data.validation import validate_customers
from telco.data.versioning import DataVersionManager


def test_generator_schema_and_churn_rate():
    df = generate_customers(5000, seed=1, target_churn_rate=0.2)
    assert list(df.columns) == CUSTOMER_COLUMNS
    assert df["customer_id"].is_unique
    assert 0.17 < df["churn"].mean() < 0.23
    assert df["nps_score"].between(0, 10).all()


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_customers(500, seed=3), generate_customers(500, seed=3))


def test_campaign_label_survives_csv_roundtrip(tmp_path):
    p = tmp_path / "c.csv"
    generate_customers(1000, seed=2).to_csv(p, index=False)
    back = pd.read_csv(p)
    assert back["campaign_type"].notna().all()
    assert "No Campaign" in set(back["campaign_type"])


def test_drift_increases_churn():
    base = generate_customers(4000, seed=5, drift="none")["churn"].mean()
    severe = generate_customers(4000, seed=5, drift="severe")["churn"].mean()
    assert severe > base + 0.1


def test_campaigns_reduce_churn():
    df = generate_customers(20000, seed=4)
    rates = df.groupby("campaign_type")["churn"].mean()
    assert rates["Discount"] < rates["No Campaign"]


def test_monthly_kpis():
    k = generate_monthly_kpis("2021-10", 60)
    assert len(k) == 60 and k["is_lebaran"].sum() == 5
    assert k["active_subscribers"].iloc[-1] > k["active_subscribers"].iloc[0]


def test_validation(small_cfg):
    df = generate_customers(300, seed=1)
    assert validate_customers(df, small_cfg).ok
    bad = df.drop(columns=["arpu_idr"])
    assert not validate_customers(bad, small_cfg).ok
    bad2 = df.assign(churn=2)
    assert not validate_customers(bad2, small_cfg).ok
    warn = df.assign(package="Gold")
    rep = validate_customers(warn, small_cfg)
    assert rep.ok and any("package" in w for w in rep.warnings)


def test_data_versioning_is_idempotent(tmp_path):
    p = tmp_path / "d.csv"
    generate_customers(200, seed=1).to_csv(p, index=False)
    dvm = DataVersionManager(tmp_path / "m.json")
    v1 = dvm.register(p, "customers")
    assert dvm.register(p, "customers")["version_id"] == v1["version_id"]
    generate_customers(200, seed=2).to_csv(p, index=False)
    assert dvm.register(p, "customers")["version"] == 2
