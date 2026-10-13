FROM node:20-alpine AS build
WORKDIR /app

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

ARG VITE_MAPBOX_TOKEN
ENV VITE_MAPBOX_TOKEN=$VITE_MAPBOX_TOKEN

ARG VITE_BASE_PATH=/
# VITE_BASE_PATH se pasa directo al CLI, no via process.env

COPY package*.json ./
RUN npm ci --legacy-peer-deps
COPY . .
RUN npm run build -- --base=${VITE_BASE_PATH}

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
