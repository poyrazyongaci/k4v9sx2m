# Shopping Sessions

Plain Python pipeline for the Shopping Sessions track.

    docker buildx build --platform linux/amd64 --load -t shopping-sessions:latest .

The container reads `/data` and writes `/output/user_daily/` and
`/output/campaign_daily/` as Parquet.
