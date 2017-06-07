
# Гистограмма количества предыдущих заказов для всех ордеров (покупателей)
orders %>%
  group_by(user_id) %>%
  summarize(Norder=n()) %>%
  arrange(desc(Norder)) %>%
  ggplot(aes(Norder))+geom_histogram() 

# Гистограмма количества предыдущих заказов для тестовых ордеров (покупателей)
order_test  <- filter(orders,eval_set=="test") %>%
  select(user_id)

order_Ntest <- filter(orders,user_id %in% order_test$user_id) %>%
  group_by(user_id) %>%
  summarize(Norder=n()) %>%
  arrange(desc(Norder)) %>%
  ggplot(aes(Norder))+geom_histogram() 
order_Ntest

# Оно одинаковое, в большинстве случаем предистория короткая

