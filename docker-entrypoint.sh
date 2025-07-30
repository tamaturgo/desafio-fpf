
mkdir -p /app/uploads
mkdir -p /app/qr_crops  
mkdir -p /app/outputs/processed_images
mkdir -p /app/logs

chmod 755 /app/uploads
chmod 755 /app/qr_crops
chmod 755 /app/outputs
chmod 755 /app/outputs/processed_images  
chmod 755 /app/logs

exec "$@"
