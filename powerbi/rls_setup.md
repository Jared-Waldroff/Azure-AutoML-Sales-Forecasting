# Row-Level Security (RLS) Setup Guide

## What is Row-Level Security?

RLS restricts data access at the row level based on who is viewing the report.
In a real deployment:
- A **Highland region manager** only sees Highland stores
- A **Coastal region manager** only sees Coastal stores
- An **executive** sees all regions

Even as a demo, implementing RLS shows employers you understand
data governance and enterprise BI requirements.

## Step 1: Create Security Roles

In Power BI Desktop:

1. Go to **Modeling** → **Manage Roles**
2. Create these roles:

### Role: Highland Region
```dax
// This DAX filter is applied to dim_store automatically.
// When active, it filters ALL fact tables connected to dim_store —
// so fact_sales and fact_forecasts both get filtered.
[region] = "Highland"
```

### Role: Coastal Region
```dax
[region] = "Coastal"
```

### Role: Amazon Region
```dax
[region] = "Amazon"
```

### Role: All Regions (Executive)
```dax
// No filter expression needed — this role sees everything.
// In Power BI Service, assign this to executive users.
```

## Step 2: Apply Filters to Correct Table

For each role (Highland, Coastal, Amazon):
1. Select the role in Manage Roles
2. Click on `dim_store` table
3. Enter the DAX filter expression
4. Click the checkmark to validate

**Important:** The filter goes on `dim_store`, NOT on fact tables.
Because dim_store has 1:* relationships to both fact tables, the filter
propagates automatically through the star schema.

## Step 3: Test RLS Locally

1. Go to **Modeling** → **View as Roles**
2. Check one of the region roles (e.g., "Highland Region")
3. Click **OK**
4. Verify that:
   - All visuals only show Highland region data
   - Card totals are lower than the unfiltered view
   - Store slicers only show Highland stores
5. Click **Stop viewing as role** when done

## Step 4: Deploy to Power BI Service

When you publish to Power BI Service:

1. **Publish** the report to a workspace
2. Go to the **dataset settings** in the workspace
3. Click **Security** tab
4. Add users/groups to each role:
   - Highland Region: highland-team@company.com
   - Coastal Region: coastal-team@company.com
   - All Regions: executives@company.com

### RLS with Azure AD Groups (Best Practice)
In production, map roles to Azure AD security groups:
- Create a group per region
- Add/remove users from the group to manage access
- Scales better than individual user assignments

## How RLS Flows Through the Star Schema

```
User logs into Power BI Service
    │
    ▼
Power BI checks user's role assignment
    │
    ▼
RLS filter applies to dim_store:
    [region] = "Highland"
    │
    ▼ (filter propagates through relationships)
    │
    ├──→ fact_sales: Only rows where store_key
    │    matches Highland stores are visible
    │
    └──→ fact_forecasts: Same filtering —
         forecasts also scoped to Highland
```

## Demo Talking Points for Employer

When presenting RLS:

1. **"I implemented Row-Level Security to demonstrate data governance."**
   - Shows you think about who should see what data

2. **"The filter is on the dimension table, not the fact table."**
   - Shows you understand star schema filter propagation

3. **"In production, I'd map these roles to Azure AD groups."**
   - Shows you know how enterprise deployments work

4. **"RLS works with both actuals and forecasts because they share the same dimension keys."**
   - Shows your schema design was intentional

## Regional Store Distribution

| Region | States | # Stores | Description |
|--------|--------|----------|-------------|
| Highland | Pichincha, Cotopaxi, Chimborazo, etc. | ~25 | Andean mountain region |
| Coastal | Guayas, Manabi, El Oro, etc. | ~22 | Pacific coast region |
| Amazon | Pastaza, Orellana | ~4 | Amazon basin region |
| Other | Unclassified | ~3 | Miscellaneous |
