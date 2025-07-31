# utils.py

def get_metric_rules():
    """
    This dictionary defines default aggregation and the "goodness" direction for each metric.
    'agg': 'sum' or 'mean'
    'is_good_when_low': True if a lower value is better (e.g., complaints, bad orders)
    """
    return {
        # --- Core Sales & Orders ---
        'Sales (Rs)': {'agg': 'sum', 'is_good_when_low': False},
        'Delivered orders': {'agg': 'sum', 'is_good_when_low': False},
        'Average order value (Rs)': {'agg': 'mean', 'is_good_when_low': False},
        'Placed Orders': {'agg': 'sum', 'is_good_when_low': False},

        # --- Quality & Complaints ---
        'Average rating': {'agg': 'mean', 'is_good_when_low': False},
        'Bad orders': {'agg': 'sum', 'is_good_when_low': True},
        'Rejected orders': {'agg': 'sum', 'is_good_when_low': True},
        'KPT+10 delayed orders': {'agg': 'sum', 'is_good_when_low': True},
        'Poor rated orders': {'agg': 'sum', 'is_good_when_low': True},
        'Total complaints': {'agg': 'sum', 'is_good_when_low': True},
        'Customer refunded complaints': {'agg': 'sum', 'is_good_when_low': True},
        'Non-refunded complaints': {'agg': 'sum', 'is_good_when_low': True},
        'Total complaints - Poor quality': {'agg': 'sum', 'is_good_when_low': True},
        'Total complaints - Poor packaging': {'agg': 'sum', 'is_good_when_low': True},
        'Total complaints - Wrong order': {'agg': 'sum', 'is_good_when_low': True},
        'Total complaints - Missing items': {'agg': 'sum', 'is_good_when_low': True},
        'Lost sales (Rs)': {'agg': 'sum', 'is_good_when_low': True},

        # --- Operational Metrics ---
        'Online %': {'agg': 'mean', 'is_good_when_low': False},
        'Offline time (in hours)': {'agg': 'sum', 'is_good_when_low': True},
        'KPT (in minutes)': {'agg': 'mean', 'is_good_when_low': True},
        'FOR accuracy (%)': {'agg': 'mean', 'is_good_when_low': False},

        # --- Funnel Metrics ---
        'Impressions': {'agg': 'sum', 'is_good_when_low': False},
        'Impressions to menu (%)': {'agg': 'mean', 'is_good_when_low': False},
        'Menu opens': {'agg': 'sum', 'is_good_when_low': False},
        'Menu to cart (%)': {'agg': 'mean', 'is_good_when_low': False},
        'Cart builds': {'agg': 'sum', 'is_good_when_low': False},
        'Cart to orders (%)': {'agg': 'mean', 'is_good_when_low': False},

        # --- User Segments & Time of Day ---
        'New user orders': {'agg': 'sum', 'is_good_when_low': False},
        'Repeat user orders': {'agg': 'sum', 'is_good_when_low': False},
        'Lapsed user orders': {'agg': 'sum', 'is_good_when_low': True},
        'Breakfast orders': {'agg': 'sum', 'is_good_when_low': False},
        'Lunch Orders': {'agg': 'sum', 'is_good_when_low': False},
        'Snack Orders': {'agg': 'sum', 'is_good_when_low': False},
        'Dinner Orders': {'agg': 'sum', 'is_good_when_low': False},
        'Late night Orders': {'agg': 'sum', 'is_good_when_low': False},

        # --- Ads & Offers ---
        'Sales from ads (Rs)': {'agg': 'sum', 'is_good_when_low': False},
        'Ads CTR (%)': {'agg': 'mean', 'is_good_when_low': False},
        'Ads impressions': {'agg': 'sum', 'is_good_when_low': False},
        'Ads menu opens': {'agg': 'sum', 'is_good_when_low': False},
        'Ads orders': {'agg': 'sum', 'is_good_when_low': False},
        'Ads spend (Rs)': {'agg': 'sum', 'is_good_when_low': True},
        'Ads ROI': {'agg': 'mean', 'is_good_when_low': False},
        'Gross sales from offers (Rs)': {'agg': 'sum', 'is_good_when_low': False},
        'Orders with offers': {'agg': 'sum', 'is_good_when_low': False},
        'Discount given (Rs)': {'agg': 'sum', 'is_good_when_low': False},
        'Effective discount (%)': {'agg': 'mean', 'is_good_when_low': True},
        'Market share (%)': {'agg': 'mean', 'is_good_when_low': False},
        'Rated orders': {'agg': 'sum', 'is_good_when_low': False}
    }

