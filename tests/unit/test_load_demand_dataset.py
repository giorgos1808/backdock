import pandas as pd
import pytest
from scripts.load_demand_dataset import SUPPLIER_COUNT, SUPPLIER_NAMES, reshape, stable_unit_price, supplier_for


def daily(store, product, category, start, days, amount=1.0):
    """One row per day, which is how the source dataset is shaped."""
    return pd.DataFrame(
        {
            "store_id": store,
            "product_id": product,
            "third_category_id": category,
            "dt": pd.date_range(start, periods=days, freq="D"),
            "sale_amount": amount,
        }
    )


class TestPeriodAggregation:
    def test_a_full_week_sums_into_one_order_line(self):
        # 2024-04-02 is a Tuesday; four full Tue-Mon weeks.
        df = daily(0, 1, 10, "2024-04-02", 28, amount=2.0)

        out, stores = reshape(df, stores=1, min_periods=1, period="weekly")

        assert stores == [0]
        assert len(out) == 4
        assert (out["sale_amount"] == 14.0).all()


    def test_incomplete_periods_at_the_edges_are_dropped(self):
        """The source range starts and ends mid-week. Those partial buckets
        sum to a fraction of a real order and would teach the model that
        demand collapses in the most recent period — the one it has to
        predict.
        """
        df = daily(0, 1, 10, "2024-04-04", 22, amount=1.0)

        out, _ = reshape(df, stores=1, min_periods=1, period="weekly")

        assert len(out) == 2
        assert (out["sale_amount"] == 7.0).all()


    def test_a_30_day_month_is_not_dropped_next_to_a_31_day_one(self):
        """Regression: measuring completeness against the longest bucket seen
        discarded every 30-day month whenever a 31-day month was present.
        """
        df = daily(0, 1, 10, "2024-04-01", 61, amount=1.0)  

        out, _ = reshape(df, stores=1, min_periods=1, period="monthly")

        assert sorted(out["sale_amount"].tolist()) == [30.0, 31.0]


    def test_monthly_period_is_supported(self):
        df = daily(0, 1, 10, "2024-04-01", 61, amount=1.0)

        out, _ = reshape(df, stores=1, min_periods=1, period="monthly")

        assert len(out) == 2
        assert sorted(out["sale_amount"].tolist()) == [30.0, 31.0]


class TestFiltering:
    def test_products_with_too_little_history_are_dropped(self):
        keep = daily(0, 1, 10, "2024-04-02", 28)
        drop = daily(0, 2, 10, "2024-04-02", 28)
        
        drop.loc[drop["dt"] >= "2024-04-09", "sale_amount"] = 0.0

        out, _ = reshape(pd.concat([keep, drop]), stores=1, min_periods=4, period="weekly")

        assert set(out["product_id"]) == {1}


    def test_zero_demand_periods_produce_no_order_line(self):
        """You don't order zero of something, you leave it off the order."""
        df = daily(0, 1, 10, "2024-04-02", 28, amount=1.0)
        df.loc[(df["dt"] >= "2024-04-09") & (df["dt"] < "2024-04-16"), "sale_amount"] = 0.0

        out, _ = reshape(df, stores=1, min_periods=1, period="weekly")

        assert len(out) == 3
        assert (out["sale_amount"] > 0).all()


    def test_a_product_that_never_sold_is_dropped_entirely(self):
        sells = daily(0, 1, 10, "2024-04-02", 28, amount=1.0)
        never = daily(0, 2, 10, "2024-04-02", 28, amount=0.0)
                   
        out, _ = reshape(pd.concat([sells, never]), stores=1, min_periods=1, period="weekly")
           
        assert set(out["product_id"]) == {1}


    def test_only_the_requested_number_of_stores_is_kept(self):
        frames = [daily(s, 1, 10, "2024-04-02", 28) for s in (0, 1, 2)]
 
        out, stores = reshape(pd.concat(frames), stores=2, min_periods=1, period="weekly")

        assert stores == [0, 1]
        assert set(out["store_id"]) == {0, 1}


class TestSupplierAssignment:
    def test_supplier_index_is_derived_from_category(self):
        df = daily(0, 1, 10, "2024-04-02", 28)
           
        out, _ = reshape(df, stores=1, min_periods=1, period="weekly")     

        assert (out["supplier_index"] == 10 % SUPPLIER_COUNT).all()   


    def test_every_supplier_index_maps_to_a_name_and_weekday(self):
        for index in range(SUPPLIER_COUNT):
            name, weekday = supplier_for(index)
            assert name in SUPPLIER_NAMES
            assert 0 <= weekday <= 6


    def test_suppliers_deliver_on_different_weekdays(self):
        """data_prep.py derives a day_of_week feature. If every order landed
        on the same weekday it would be constant and carry no signal.
        """
        weekdays = {supplier_for(i)[1] for i in range(SUPPLIER_COUNT)}

        assert len(weekdays) > 1


class TestUnitPrice:
    def test_is_stable_for_a_given_product(self):
        assert stable_unit_price(42) == stable_unit_price(42)


    def test_differs_across_products(self):
        prices = {stable_unit_price(i) for i in range(50)}

        assert len(prices) > 25


    @pytest.mark.parametrize("product_id", [0, 1, 999, 123456])
    def test_is_a_plausible_price(self, product_id):
        assert 0.80 <= stable_unit_price(product_id) <= 42.80
